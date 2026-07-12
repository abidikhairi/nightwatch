from dataclasses import dataclass


@dataclass
class ModelSummary:
    repo_id: str
    downloads: int
    likes: int
    pipeline_tag: str | None


@dataclass
class ModelDetails:
    repo_id: str
    num_params: int | None
    size_on_disk_bytes: int
    dtype: str | None
    max_context_length: int | None
    has_chat_template: bool
    supports_tools: bool
    quantization_config: dict | None
    hidden_size: int | None
    num_hidden_layers: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None


@dataclass
class Recommendation:
    num_users: int
    estimated_memory_gb: float
    gpu_recommendation: str
    quantization_advice: str | None
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_num_seqs: int
    max_model_len: int | None
    dtype_flag: str
    quantization_flag: str | None
    serve_command: str
