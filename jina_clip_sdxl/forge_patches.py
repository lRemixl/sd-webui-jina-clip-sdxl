import math
import types

import torch


_patch_state = {
    "installed": False,
    "orig_compile_conditions": None,
    "orig_sampling_compile_conditions": None,
    "orig_cross_attention_forward": None,
    "orig_basic_transformer_forward": None,
}


def _repeat_to_batch_size(tensor, batch_size):
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    if tensor.shape[0] < batch_size:
        return tensor.repeat([math.ceil(batch_size / tensor.shape[0])] + [1] * (len(tensor.shape) - 1))[:batch_size]
    return tensor


def _mask_to_additive(mask, x, context):
    if mask is None or context is None:
        return None

    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        return None
    if mask.shape[-1] != context.shape[1]:
        return None

    batch_size, query_len = x.shape[:2]
    if mask.shape[0] != batch_size and mask.shape[0] > 0:
        mask = mask.repeat(math.ceil(batch_size / mask.shape[0]), 1)[:batch_size]

    mask = mask.to(device=x.device, dtype=x.dtype)
    additive = (1.0 - mask) * -10000.0
    return additive.view(mask.shape[0], 1, 1, mask.shape[1]).expand(mask.shape[0], 1, query_len, mask.shape[1])


def _split_context_mask(x, context, attn):
    if context is None or attn is None or not hasattr(attn, "to_k"):
        return context, None

    expected_dim = getattr(attn.to_k, "in_features", None)
    if expected_dim is None or context.shape[-1] != expected_dim + 1:
        return context, None

    mask = context[..., -1]
    context = context[..., :-1]
    return context, _mask_to_additive(mask, x, context)


def _merge_attention_mask(args, kwargs, additive_mask):
    if len(args) > 3:
        new_args = list(args)
        new_args[3] = additive_mask if new_args[3] is None else new_args[3] + additive_mask
        return tuple(new_args), kwargs

    new_kwargs = dict(kwargs)
    existing_mask = new_kwargs.get("mask", None)
    new_kwargs["mask"] = additive_mask if existing_mask is None else existing_mask + additive_mask
    return args, new_kwargs


class ConditionJinaMask:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return ConditionJinaMask(cond)

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(_repeat_to_batch_size(self.cond, batch_size).to(device))

    def can_concat(self, other):
        s1 = self.cond.shape
        s2 = other.cond.shape
        if s1 == s2:
            return True
        if len(s1) != len(s2) or len(s1) < 2:
            return False
        if s1[0] != s2[0] or s1[-1] != s2[-1]:
            return False
        return math.lcm(s1[1], s2[1]) // min(s1[1], s2[1]) <= 4

    def concat(self, others):
        conds = [self.cond] + [x.cond for x in others]
        max_len = conds[0].shape[1]
        for cond in conds[1:]:
            max_len = math.lcm(max_len, cond.shape[1])

        out = []
        for cond in conds:
            if cond.shape[1] < max_len:
                cond = cond.repeat(1, max_len // cond.shape[1], 1)
            out.append(cond)
        return torch.cat(out)


def install_global_patches():
    if _patch_state["installed"]:
        return

    from backend.nn import unet as unet_module
    from backend.sampling import condition as condition_module
    from backend.sampling import sampling_function as sampling_module

    orig_compile_conditions = condition_module.compile_conditions
    orig_sampling_compile_conditions = sampling_module.compile_conditions
    orig_cross_attention_forward = unet_module.CrossAttention.forward
    orig_basic_transformer_forward = unet_module.BasicTransformerBlock.forward

    def compile_conditions_with_jina_mask(cond):
        result = orig_compile_conditions(cond)
        if isinstance(cond, dict) and "jina_attention_mask" in cond:
            for entry in result:
                entry["model_conds"]["jina_attention_mask"] = ConditionJinaMask(cond["jina_attention_mask"])
        return result

    def cross_attention_forward_with_jina_mask(self, x, context=None, value=None, mask=None, transformer_options={}):
        context, inline_additive = _split_context_mask(x, context, self)
        if context is not None and value is not None and value.shape[-1] == context.shape[-1] + 1:
            value = value[..., :-1]
        if inline_additive is not None:
            mask = inline_additive if mask is None else mask + inline_additive

        jina_mask = None
        if isinstance(transformer_options, dict):
            jina_mask = transformer_options.get("jina_cross_attention_mask", None)
        additive = _mask_to_additive(jina_mask, x, context)
        if additive is not None:
            mask = additive if mask is None else mask + additive
        return orig_cross_attention_forward(self, x, context=context, value=value, mask=mask, transformer_options=transformer_options)

    def basic_transformer_forward_with_jina_mask(self, x, context=None, transformer_options={}):
        context, additive = _split_context_mask(x, context, getattr(self, "attn2", None))
        if additive is None:
            return orig_basic_transformer_forward(self, x, context=context, transformer_options=transformer_options)

        original_attn2_forward = self.attn2.forward

        def wrapped_attn2_forward(*args, **kwargs):
            args, kwargs = _merge_attention_mask(args, kwargs, additive)
            return original_attn2_forward(*args, **kwargs)

        self.attn2.forward = wrapped_attn2_forward
        try:
            return orig_basic_transformer_forward(self, x, context=context, transformer_options=transformer_options)
        finally:
            self.attn2.forward = original_attn2_forward

    condition_module.compile_conditions = compile_conditions_with_jina_mask
    sampling_module.compile_conditions = compile_conditions_with_jina_mask
    unet_module.BasicTransformerBlock.forward = basic_transformer_forward_with_jina_mask
    unet_module.CrossAttention.forward = cross_attention_forward_with_jina_mask

    _patch_state.update(
        installed=True,
        orig_compile_conditions=orig_compile_conditions,
        orig_sampling_compile_conditions=orig_sampling_compile_conditions,
        orig_cross_attention_forward=orig_cross_attention_forward,
        orig_basic_transformer_forward=orig_basic_transformer_forward,
    )


def install_unet_mask_wrapper(unet):
    if unet is None:
        return

    model_options = getattr(unet, "model_options", None)
    if model_options is None:
        return

    existing = model_options.get("model_function_wrapper", None)
    if getattr(existing, "_jina_mask_wrapper", False):
        return

    def wrapper(apply_model, params):
        c = params.get("c", {})
        mask = c.get("jina_attention_mask", None) if isinstance(c, dict) else None
        if mask is not None:
            c = c.copy()
            c.pop("jina_attention_mask", None)
            transformer_options = c.get("transformer_options", {})
            transformer_options = transformer_options.copy() if isinstance(transformer_options, dict) else {}
            transformer_options["jina_cross_attention_mask"] = mask
            c["transformer_options"] = transformer_options
            params = params.copy()
            params["c"] = c

        if existing is not None:
            return existing(apply_model, params)

        return apply_model(params["input"], params["timestep"], **params["c"])

    wrapper._jina_mask_wrapper = True
    wrapper._jina_previous_wrapper = existing
    unet.set_model_unet_function_wrapper(wrapper)


def patch_model_conditioning(engine, cfg, manager):
    if getattr(engine, "_jina_original_get_learned_conditioning", None) is None:
        engine._jina_original_get_learned_conditioning = engine.get_learned_conditioning

    def patched_get_learned_conditioning(prompts):
        return manager.get_conditioning(engine, prompts, cfg)

    engine.get_learned_conditioning = patched_get_learned_conditioning


def restore_model_conditioning(engine):
    original = getattr(engine, "_jina_original_get_learned_conditioning", None)
    if original is not None:
        engine.get_learned_conditioning = original
        engine._jina_original_get_learned_conditioning = None
