import logging
import os

import gradio as gr

from modules import scripts, shared

from jina_clip_sdxl.attention_override import apply_sage_override, restore_sage_override
from jina_clip_sdxl.encoder import JinaConfig, manager
from jina_clip_sdxl.forge_patches import (
    install_global_patches,
    install_unet_mask_wrapper,
    patch_model_conditioning,
    restore_model_conditioning,
)
from jina_clip_sdxl.mugen_loader_patch import install_mugen_loader_patch
from jina_clip_sdxl.paths import (
    default_adapter_path,
    default_jina_model_path,
    find_adapters,
    find_jina_models,
    resolve_adapter_path,
    resolve_jina_model_path,
)

logger = logging.getLogger("JinaCLIP-SDXL")

install_mugen_loader_patch()


def _is_mugen_like(model):
    if model is None:
        return False
    if model.__class__.__name__.lower() == "mugen":
        return True
    return bool(getattr(model, "is_sdxl", False) and hasattr(model, "embedder"))


def _build_config(
    model_id,
    adapter_path,
    device_name,
    dtype_name,
    max_length,
    padding_mode,
    weighting_mode,
    adapter_seq_len,
    attn_pooling,
    use_positional,
    format_tags,
    cross_attention_mask,
    local_files_only,
    convert_legacy_mha,
):
    return JinaConfig(
        model_id=resolve_jina_model_path(model_id),
        adapter_path=resolve_adapter_path(adapter_path),
        device_name=device_name,
        dtype_name=dtype_name,
        max_length=int(max_length),
        padding_mode=padding_mode,
        weighting_mode=weighting_mode,
        adapter_seq_len=int(adapter_seq_len),
        attn_pooling=bool(attn_pooling),
        use_positional=bool(use_positional),
        format_tags=bool(format_tags),
        cross_attention_mask=bool(cross_attention_mask),
        local_files_only=bool(local_files_only),
        convert_legacy_mha=bool(convert_legacy_mha),
    )


class JinaClipSDXLScript(scripts.Script):
    sorting_priority = 20

    def title(self):
        return "Jina CLIP v2 SDXL Adapter"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Jina CLIP v2 SDXL Adapter", open=False):
            enabled = gr.Checkbox(False, label="Enabled")
            model_choices = find_jina_models()
            model_default = default_jina_model_path()
            if model_default and model_default not in model_choices:
                model_choices = [model_default] + model_choices
            adapter_choices = find_adapters()
            adapter_default = default_adapter_path()
            if adapter_default and adapter_default not in adapter_choices:
                adapter_choices = [adapter_default] + adapter_choices

            model_id = gr.Dropdown(
                label="Jina model",
                choices=model_choices,
                value=model_default,
                allow_custom_value=True,
            )
            adapter_path = gr.Dropdown(
                label="Adapter .safetensors",
                choices=adapter_choices,
                value=adapter_default,
                allow_custom_value=True,
            )
            with gr.Row():
                device_name = gr.Dropdown(
                    label="Device",
                    choices=["auto", "cuda:0", "cuda:1", "cpu"],
                    value="auto",
                )
                dtype_name = gr.Dropdown(
                    label="Jina dtype",
                    choices=["auto", "bf16", "fp16", "fp32"],
                    value="auto",
                )
                max_length = gr.Dropdown(
                    label="Tokenizer max length",
                    choices=["512", "1024"],
                    value="1024",
                )
            with gr.Row():
                padding_mode = gr.Dropdown(
                    label="Padding",
                    choices=["none", "Nearest-77", "539", "1078"],
                    value="Nearest-77 Chunk",
                )
                adapter_seq_len = gr.Dropdown(
                    label="Adapter max seq len",
                    choices=["539", "1078"],
                    value="1078",
                )
                weighting_mode = gr.Dropdown(
                    label="Prompt weighting",
                    choices=["comfy", "A1111", "skip"],
                    value="comfy",
                )
            with gr.Row():
                attn_pooling = gr.Checkbox(True, label="Attention pooled vector")
                use_positional = gr.Checkbox(False, label="Use positional embeddings")
                format_tags = gr.Checkbox(True, label="Format @ tags")
            with gr.Row():
                cross_attention_mask = gr.Checkbox(True, label="Cross-attention padding mask")
                sage_attention_override = gr.Dropdown(
                    label="SageAttention override",
                    choices=["off", "fp16_triton"],
                    value="fp16_triton",
                )
                local_files_only = gr.Checkbox(False, label="Local files only")
                convert_legacy_mha = gr.Checkbox(False, label="Convert legacy MHA adapter")

        self.infotext_fields = [
            (enabled, "Jina CLIP v2"),
            (model_id, "Jina model"),
            (adapter_path, "Jina adapter"),
            (padding_mode, "Jina padding"),
            (weighting_mode, "Jina weighting"),
            (sage_attention_override, "Jina SageAttention override"),
        ]

        return [
            enabled,
            model_id,
            adapter_path,
            device_name,
            dtype_name,
            max_length,
            padding_mode,
            weighting_mode,
            adapter_seq_len,
            attn_pooling,
            use_positional,
            format_tags,
            cross_attention_mask,
            sage_attention_override,
            local_files_only,
            convert_legacy_mha,
        ]

    def _activate(self, p, *args):
        enabled = bool(args[0])
        model = shared.sd_model

        if not enabled:
            if model is not None:
                restore_model_conditioning(model)
            restore_sage_override()
            return None

        if not _is_mugen_like(model):
            raise RuntimeError("Jina CLIP v2 SDXL Adapter expects a loaded Mugen/SDXL-like checkpoint.")

        cfg = _build_config(*args[1:13], *args[14:])
        if not cfg.model_id:
            raise ValueError("Enter a Jina model path or Hugging Face id.")

        sage_attention_override = args[13]
        apply_sage_override(sage_attention_override)
        install_global_patches()
        patch_model_conditioning(model, cfg, manager)
        install_unet_mask_wrapper(getattr(model.forge_objects, "unet", None))
        install_unet_mask_wrapper(getattr(model.forge_objects_after_applying_lora, "unet", None))

        p.clear_prompt_cache()
        p.extra_generation_params["Jina CLIP v2"] = "enabled"
        p.extra_generation_params["Jina model"] = os.path.basename(cfg.model_id.rstrip("\\/")) or cfg.model_id
        p.extra_generation_params["Jina adapter"] = os.path.basename(cfg.adapter_path)
        p.extra_generation_params["Jina padding"] = cfg.padding_mode
        p.extra_generation_params["Jina weighting"] = cfg.weighting_mode
        if sage_attention_override != "off":
            p.extra_generation_params["Jina SageAttention override"] = sage_attention_override
        return cfg

    def process(self, p, *args):
        self._activate(p, *args)

    def process_batch(self, p, *args, **kwargs):
        cfg = self._activate(p, *args)
        if cfg is not None:
            setattr(p, "_jina_clip_sdxl_cfg", cfg)

    def process_before_every_sampling(self, p, *args, **kwargs):
        if not bool(args[0]):
            return
        model = shared.sd_model
        if model is None:
            return
        install_unet_mask_wrapper(getattr(model.forge_objects, "unet", None))
        install_unet_mask_wrapper(getattr(model.forge_objects_after_applying_lora, "unet", None))

    def postprocess(self, p, processed, *args):
        model = shared.sd_model
        if model is not None:
            restore_model_conditioning(model)
        restore_sage_override()
