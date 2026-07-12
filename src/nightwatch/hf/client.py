import json

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from nightwatch.hf.models import ModelDetails, ModelSummary

_api = HfApi()

_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin")


def search_models(query: str, limit: int = 20) -> list[ModelSummary]:
    if not query.strip():
        return []

    results = _api.list_models(search=query, limit=limit, sort="downloads")
    return [
        ModelSummary(
            repo_id=model.id,
            downloads=model.downloads or 0,
            likes=model.likes or 0,
            pipeline_tag=model.pipeline_tag,
        )
        for model in results
    ]


def _download_json(repo_id: str, filename: str) -> dict | None:
    try:
        path = hf_hub_download(repo_id, filename)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    with open(path) as f:
        return json.load(f)


def get_model_details(repo_id: str) -> ModelDetails:
    info = _api.model_info(repo_id, files_metadata=True)

    num_params = info.safetensors.total if info.safetensors else None

    size_on_disk_bytes = sum(
        sibling.size or 0
        for sibling in info.siblings or []
        if sibling.rfilename.endswith(_WEIGHT_FILE_SUFFIXES)
    )

    config = _download_json(repo_id, "config.json") or {}
    tokenizer_config = _download_json(repo_id, "tokenizer_config.json") or {}

    chat_template = tokenizer_config.get("chat_template")
    has_chat_template = chat_template is not None
    supports_tools = bool(chat_template) and (
        "tool_call" in chat_template or "<tools>" in chat_template
    )

    return ModelDetails(
        repo_id=repo_id,
        num_params=num_params,
        size_on_disk_bytes=size_on_disk_bytes,
        dtype=config.get("torch_dtype"),
        max_context_length=config.get("max_position_embeddings"),
        has_chat_template=has_chat_template,
        supports_tools=supports_tools,
        quantization_config=config.get("quantization_config"),
        hidden_size=config.get("hidden_size"),
        num_hidden_layers=config.get("num_hidden_layers"),
        num_attention_heads=config.get("num_attention_heads"),
        num_key_value_heads=config.get("num_key_value_heads") or config.get("num_attention_heads"),
    )
