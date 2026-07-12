from nightwatch.hf.models import ModelDetails, Recommendation


def _format_bytes(num_bytes: int) -> str:
    gb = num_bytes / 1_000_000_000
    return f"{gb:.2f} GB"


def format_model_facts_markdown(details: ModelDetails) -> str:
    params = f"{details.num_params:,}" if details.num_params is not None else "unknown"
    dtype = details.dtype or "unknown"
    context = f"{details.max_context_length:,} tokens" if details.max_context_length else "unknown"
    chat_template = "yes" if details.has_chat_template else "no"
    tool_support = "yes" if details.supports_tools else "no"

    if details.quantization_config:
        method = details.quantization_config.get("quant_method", "unknown")
        bits = details.quantization_config.get("bits", "?")
        quantization = f"{method}, {bits}-bit"
    else:
        quantization = "none (full precision)"

    lines = [
        f"# {details.repo_id}",
        "",
        f"- **Parameters**: {params}",
        f"- **Size on disk**: {_format_bytes(details.size_on_disk_bytes)}",
        f"- **Dtype**: {dtype}",
        f"- **Max context length**: {context}",
        f"- **Chat template**: {chat_template}",
        f"- **Tool support**: {tool_support}",
        f"- **Quantization**: {quantization}",
    ]
    return "\n".join(lines)


def format_details_markdown(details: ModelDetails, recommendation: Recommendation) -> str:
    lines = [
        format_model_facts_markdown(details),
        "",
        f"## Recommendation ({recommendation.num_users} concurrent users)",
        "",
        f"- **Estimated memory**: {recommendation.estimated_memory_gb} GB",
        f"- **GPU**: {recommendation.gpu_recommendation}",
        f"- **Tensor parallel size**: {recommendation.tensor_parallel_size}",
        f"- **GPU memory utilization**: {recommendation.gpu_memory_utilization}",
        f"- **Max concurrent sequences**: {recommendation.max_num_seqs}",
    ]
    if recommendation.quantization_advice:
        lines.append(f"- **Advice**: {recommendation.quantization_advice}")

    lines += [
        "",
        "```bash",
        recommendation.serve_command,
        "```",
    ]

    return "\n".join(lines)
