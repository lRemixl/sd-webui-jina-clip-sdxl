from contextlib import contextmanager
import json
import sys

from packaging.version import Version
import transformers


TRANSFORMERS_VERSION = Version(transformers.__version__)


def validate_jina_tokenizer(tokenizer):
    """Reject token IDs changed by Transformers' optional Mistral regex repair."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    pre_tokenizer = getattr(backend, "pre_tokenizer", None)
    get_state = getattr(pre_tokenizer, "__getstate__", None)
    if not callable(get_state):
        return
    state = get_state()
    if isinstance(state, bytes):
        state = state.decode("utf-8")
    if isinstance(state, str):
        state = json.loads(state)
    if not isinstance(state, dict):
        return
    components = state.get("pretokenizers", []) if state.get("type") == "Sequence" else [state]
    component_types = tuple(component.get("type") for component in components if isinstance(component, dict))
    if component_types and component_types[0] == "Split":
        raise RuntimeError(
            "jina-clip-v2 was loaded with Transformers' Mistral regex repair. "
            "Reload with fix_mistral_regex=False to preserve the token IDs used to train the adapter."
        )


@contextmanager
def force_local_files_only_for_transformers():
    """Propagate offline loading into Jina's nested trusted-code loads."""
    from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer
    from transformers.models.auto import auto_factory

    classes = (AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer)
    previous = []
    for auto_class in classes:
        had_override = "from_pretrained" in auto_class.__dict__
        descriptor = auto_class.__dict__.get("from_pretrained")
        original = auto_class.from_pretrained

        def local_from_pretrained(
            cls,
            *args,
            _original=original,
            _is_tokenizer=auto_class is AutoTokenizer,
            **kwargs,
        ):
            kwargs["local_files_only"] = True
            if _is_tokenizer:
                kwargs["fix_mistral_regex"] = False
            return _original(*args, **kwargs)

        previous.append((auto_class, had_override, descriptor))
        auto_class.from_pretrained = classmethod(local_from_pretrained)

    original_dynamic_loader = auto_factory.get_class_from_dynamic_module

    def local_dynamic_loader(*args, **kwargs):
        kwargs["local_files_only"] = True
        return original_dynamic_loader(*args, **kwargs)

    auto_factory.get_class_from_dynamic_module = local_dynamic_loader
    try:
        yield
    finally:
        auto_factory.get_class_from_dynamic_module = original_dynamic_loader
        for auto_class, had_override, descriptor in reversed(previous):
            if had_override:
                auto_class.from_pretrained = descriptor
            else:
                delattr(auto_class, "from_pretrained")


def pretrained_dtype_kwargs(dtype):
    if TRANSFORMERS_VERSION.major >= 5:
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def ensure_legacy_clip_symbols():
    """Supply CLIP helpers expected by Jina's trusted code when HF removes them."""
    from transformers.models.clip import modeling_clip

    if hasattr(modeling_clip, "clip_loss"):
        return

    import torch
    import torch.nn.functional as functional

    def contrastive_loss(logits):
        labels = torch.arange(len(logits), device=logits.device)
        return functional.cross_entropy(logits, labels)

    def clip_loss(similarity):
        return (contrastive_loss(similarity) + contrastive_loss(similarity.t())) / 2.0

    modeling_clip.clip_loss = clip_loss


def ensure_flash_attn_bypass():
    try:
        from flash_attn.ops.triton.rotary import apply_rotary  # noqa: F401
        return
    except Exception:
        pass
    for module_name in list(sys.modules):
        if module_name.startswith("flash_attn"):
            del sys.modules[module_name]


@contextmanager
def ignore_none_code_revision_for_auto_model_from_config():
    from transformers import AutoModel

    had_override = "from_config" in AutoModel.__dict__
    previous_override = AutoModel.__dict__.get("from_config")
    original_from_config = AutoModel.from_config

    def compatible_from_config(cls, config, **kwargs):
        if kwargs.get("code_revision") is None:
            kwargs.pop("code_revision", None)
        return original_from_config(config, **kwargs)

    AutoModel.from_config = classmethod(compatible_from_config)
    try:
        yield
    finally:
        if had_override:
            AutoModel.from_config = previous_override
        else:
            delattr(AutoModel, "from_config")


def repair_jina_clip_nonpersistent_buffers(model):
    """Repair Jina buffers that are initialized on meta during HF loading."""
    import torch

    repaired = 0
    masks_seen = 0
    rotary_seen = 0
    for module in model.modules():
        dropout_mask = module._buffers.get("lora_dropout_mask")
        if isinstance(dropout_mask, torch.Tensor):
            masks_seen += 1
            if not bool(torch.eq(dropout_mask, 1).all()):
                dropout_mask.fill_(1)
                repaired += 1

        inv_freq = module._buffers.get("inv_freq")
        compute_inv_freq = getattr(module, "_compute_inv_freq", None)
        if isinstance(inv_freq, torch.Tensor) and callable(compute_inv_freq):
            rotary_seen += 1
            expected = compute_inv_freq(device=inv_freq.device).to(dtype=inv_freq.dtype)
            if expected.shape != inv_freq.shape:
                raise RuntimeError(
                    "Jina rotary buffer shape mismatch: "
                    f"loaded={tuple(inv_freq.shape)}, expected={tuple(expected.shape)}"
                )
            if not torch.equal(inv_freq, expected):
                inv_freq.copy_(expected)
                repaired += 1
                for cache_name in ("_cos_cached", "_sin_cached", "_cos_k_cached", "_sin_k_cached"):
                    if hasattr(module, cache_name):
                        setattr(module, cache_name, None)
                if hasattr(module, "_seq_len_cached"):
                    module._seq_len_cached = 0

    return {"repaired": repaired, "lora_dropout_masks": masks_seen, "rotary_inv_freq": rotary_seen}
