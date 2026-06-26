import logging

logger = logging.getLogger("JinaCLIP-SDXL")

_state = {
    "active": False,
    "original_sageattn": None,
}


def apply_sage_override(mode):
    mode = (mode or "off").strip()
    if mode == "off":
        restore_sage_override()
        return

    if mode != "fp16_triton":
        raise ValueError(f"Unsupported Jina SageAttention override: {mode}")

    try:
        from backend import attention as attention_module
    except Exception as exc:
        raise RuntimeError(f"Could not import Forge attention module: {exc}") from exc

    if not hasattr(attention_module, "sageattn"):
        logger.warning("Jina requested SageAttention fp16 Triton, but SageAttention is not active in Forge.")
        return

    try:
        import sageattention
    except Exception as exc:
        raise RuntimeError(f"Could not import sageattention: {exc}") from exc

    triton_func = getattr(sageattention, "sageattn_qk_int8_pv_fp16_triton", None)
    if triton_func is None:
        logger.warning("Installed sageattention does not expose sageattn_qk_int8_pv_fp16_triton.")
        return

    if not _state["active"]:
        _state["original_sageattn"] = attention_module.sageattn

    attention_module.sageattn = triton_func
    _state["active"] = True
    logger.info("Jina CLIP v2 SDXL Adapter forced SageAttention 2 fp16 Triton for this run.")


def restore_sage_override():
    if not _state["active"]:
        return

    try:
        from backend import attention as attention_module

        if _state["original_sageattn"] is not None:
            attention_module.sageattn = _state["original_sageattn"]
    finally:
        _state["active"] = False
        _state["original_sageattn"] = None
        logger.info("Jina CLIP v2 SDXL Adapter restored Forge SageAttention function.")
