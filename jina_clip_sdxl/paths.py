import os
from pathlib import Path

from modules import paths


def _models_dir():
    return Path(paths.models_path)


def _extension_dir():
    return Path(__file__).resolve().parents[1]


def _relative_or_absolute(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def default_jina_model_path():
    models = find_jina_models()
    if models:
        return models[0]
    return "jinaai/jina-clip-v2"


def adapter_roots():
    return [
        _models_dir() / "llm_adapter",
        _models_dir() / "llm_adapters",
        _models_dir() / "LLM_Adapters",
        _models_dir() / "Jina_Adapters",
        _extension_dir() / "models" / "llm_adapter",
    ]


def jina_model_roots():
    return [
        _models_dir() / "text_encoder",
        _models_dir() / "llm",
        _models_dir() / "LLM",
        _models_dir() / "Jina",
        _extension_dir() / "models" / "LLM",
    ]


def find_adapters():
    found = []
    for root in adapter_roots():
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.safetensors")):
            found.append(_relative_or_absolute(path, root))
    return found


def _looks_like_hf_text_encoder(path):
    if not path.is_dir():
        return False
    markers = (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
        "pytorch_model.bin",
    )
    return any((path / marker).exists() for marker in markers)


def find_jina_models():
    found = []
    seen = set()
    for root in jina_model_roots():
        if not root.exists():
            continue

        candidates = [root] if _looks_like_hf_text_encoder(root) else []
        candidates.extend(path for path in sorted(root.rglob("*")) if _looks_like_hf_text_encoder(path))

        for path in candidates:
            value = _relative_or_absolute(path, root)
            if value in seen:
                continue
            seen.add(value)
            found.append(value)

    return found


def default_adapter_path():
    adapters = find_adapters()
    return adapters[0] if adapters else ""


def resolve_jina_model_path(value):
    value = (value or "").strip().strip('"')
    if not value:
        raise ValueError("Select or enter a Jina model folder or Hugging Face id.")

    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if expanded.exists():
        return str(expanded)

    for root in jina_model_roots():
        candidate = root / value
        if candidate.exists():
            return str(candidate)

    return value


def resolve_adapter_path(value):
    value = (value or "").strip().strip('"')
    if not value:
        raise ValueError("Select or enter a Jina adapter .safetensors path.")

    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if expanded.is_file():
        return str(expanded)

    for root in adapter_roots():
        candidate = root / value
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(f"Jina adapter not found: {value}")
