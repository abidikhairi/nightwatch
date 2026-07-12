import shlex

from nightwatch.hf.models import ModelDetails, Recommendation

_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
}

_OVERHEAD_FACTOR = 1.2
_KV_CACHE_DTYPE_BYTES = 2

_BYTES_PER_GB = 1_000_000_000

_SINGLE_GPU_CAPACITY_GB = 80
_DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
_DEFAULT_DTYPE_FLAG = "auto"


def _bytes_per_param(details: ModelDetails) -> float:
    quant = details.quantization_config
    if quant and quant.get("bits"):
        return quant["bits"] / 8
    return _DTYPE_BYTES.get(details.dtype or "", 2)


def _kv_cache_bytes(details: ModelDetails, num_users: int) -> int | None:
    if not (
        details.hidden_size
        and details.num_hidden_layers
        and details.num_attention_heads
        and details.num_key_value_heads
        and details.max_context_length
    ):
        return None

    head_dim = details.hidden_size / details.num_attention_heads
    kv_bytes_per_token = (
        2
        * details.num_hidden_layers
        * details.num_key_value_heads
        * head_dim
        * _KV_CACHE_DTYPE_BYTES
    )
    return int(num_users * details.max_context_length * kv_bytes_per_token)


def _gpu_tier(total_memory_gb: float) -> str:
    if total_memory_gb <= 24:
        return "single consumer GPU (RTX 4090/3090, 24GB)"
    if total_memory_gb <= 48:
        return "single prosumer/datacenter GPU (L40S/A6000, 48GB)"
    if total_memory_gb <= _SINGLE_GPU_CAPACITY_GB:
        return "single datacenter GPU (A100/H100, 80GB)"
    return "multi-GPU tensor-parallel deployment (model does not fit on a single 80GB GPU)"


def _next_power_of_two(n: int) -> int:
    power = 1
    while power < n:
        power *= 2
    return power


def _tensor_parallel_size(total_memory_gb: float) -> int:
    if total_memory_gb <= _SINGLE_GPU_CAPACITY_GB:
        return 1
    gpus_needed = -(-int(total_memory_gb) // _SINGLE_GPU_CAPACITY_GB)  # ceil division
    return _next_power_of_two(gpus_needed)


def _quantization_flag(details: ModelDetails) -> str | None:
    if details.quantization_config:
        return details.quantization_config.get("quant_method")
    return None


def _dtype_flag(details: ModelDetails) -> str:
    if details.dtype in _DTYPE_BYTES:
        return details.dtype
    return _DEFAULT_DTYPE_FLAG


def build_serve_args(
    repo_id: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    max_model_len: int | None,
    dtype_flag: str,
    quantization_flag: str | None,
) -> list[str]:
    args = ["vllm", "serve", repo_id]
    args += ["--tensor-parallel-size", str(tensor_parallel_size)]
    args += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    args += ["--max-num-seqs", str(max_num_seqs)]
    if max_model_len is not None:
        args += ["--max-model-len", str(max_model_len)]
    args += ["--dtype", dtype_flag]
    if quantization_flag is not None:
        args += ["--quantization", quantization_flag]
    return args


def build_serve_command(
    repo_id: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    max_model_len: int | None,
    dtype_flag: str,
    quantization_flag: str | None,
) -> str:
    args = build_serve_args(
        repo_id=repo_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        dtype_flag=dtype_flag,
        quantization_flag=quantization_flag,
    )
    return " ".join(shlex.quote(arg) for arg in args)


def recommend_deployment(details: ModelDetails, num_users: int = 10) -> Recommendation:
    if details.num_params is None:
        return Recommendation(
            num_users=num_users,
            estimated_memory_gb=0.0,
            gpu_recommendation="unknown (parameter count unavailable)",
            quantization_advice=None,
            tensor_parallel_size=1,
            gpu_memory_utilization=_DEFAULT_GPU_MEMORY_UTILIZATION,
            max_num_seqs=num_users,
            max_model_len=details.max_context_length,
            dtype_flag=_dtype_flag(details),
            quantization_flag=_quantization_flag(details),
            serve_command=f"vllm serve {details.repo_id}",
        )

    weight_memory_bytes = details.num_params * _bytes_per_param(details)
    kv_cache_bytes = _kv_cache_bytes(details, num_users)

    total_bytes = weight_memory_bytes * _OVERHEAD_FACTOR + (kv_cache_bytes or 0)
    total_memory_gb = total_bytes / _BYTES_PER_GB

    quantization_advice = None
    if details.quantization_config:
        method = details.quantization_config.get("quant_method", "unknown")
        bits = details.quantization_config.get("bits", "?")
        quantization_advice = f"Already quantized ({method}, {bits}-bit)."
    elif weight_memory_bytes / _BYTES_PER_GB > 24:
        quantization_advice = (
            "Consider 4-bit quantization (AWQ/GPTQ) to fit on a single consumer/prosumer GPU."
        )

    if kv_cache_bytes is None:
        quantization_advice = (
            (quantization_advice + " " if quantization_advice else "")
            + "Note: KV cache could not be estimated (missing config fields); "
            "estimate reflects model weights only."
        )

    tensor_parallel_size = _tensor_parallel_size(total_memory_gb)
    gpu_memory_utilization = _DEFAULT_GPU_MEMORY_UTILIZATION
    max_num_seqs = num_users
    max_model_len = details.max_context_length
    dtype_flag = _dtype_flag(details)
    quantization_flag = _quantization_flag(details)

    serve_command = build_serve_command(
        repo_id=details.repo_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        dtype_flag=dtype_flag,
        quantization_flag=quantization_flag,
    )

    return Recommendation(
        num_users=num_users,
        estimated_memory_gb=round(total_memory_gb, 2),
        gpu_recommendation=_gpu_tier(total_memory_gb),
        quantization_advice=quantization_advice,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        dtype_flag=dtype_flag,
        quantization_flag=quantization_flag,
        serve_command=serve_command,
    )
