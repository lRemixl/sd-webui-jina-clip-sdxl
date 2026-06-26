import logging

import torch

logger = logging.getLogger("JinaCLIP-SDXL")


def _is_mugen_guess(guess):
    return guess is not None and guess.__class__.__name__ == "Mugen"


def _missing_clip_state_dict(state_dict):
    return not (isinstance(state_dict, dict) and len(state_dict) > 16)


def install_mugen_loader_patch():
    """Allow Mugen checkpoints without bundled SDXL CLIP weights.

    Jina-adapter checkpoints do not need Forge's SDXL CLIP encoders for prompt
    conditioning, but Forge's native Mugen loader still tries to instantiate them.
    This patch keeps the existing Mugen path for normal checkpoints and creates a
    lightweight Jina-only Mugen object when CLIP components are absent.
    """

    from backend import loader
    from backend.diffusion_engine import mugen as mugen_module

    if not getattr(loader, "_jina_mugen_loader_patch_installed", False):
        original_load_component = loader.load_huggingface_component

        def load_huggingface_component_jina_mugen(guess, component_name, lib_name, cls_name, repo_path, state_dict):
            if (
                _is_mugen_guess(guess)
                and component_name.startswith("text_encoder")
                and cls_name in ("CLIPTextModel", "CLIPTextModelWithProjection")
                and _missing_clip_state_dict(state_dict)
            ):
                logger.warning(
                    "Skipping missing %s for Jina-only Mugen checkpoint. "
                    "Enable the Jina CLIP v2 SDXL Adapter extension for prompt conditioning.",
                    component_name,
                )
                return None

            return original_load_component(guess, component_name, lib_name, cls_name, repo_path, state_dict)

        loader.load_huggingface_component = load_huggingface_component_jina_mugen
        loader._jina_mugen_loader_patch_installed = True

    Mugen = mugen_module.Mugen
    if getattr(Mugen, "_jina_missing_clip_patch_installed", False):
        return

    original_init = Mugen.__init__
    original_set_clip_skip = Mugen.set_clip_skip
    original_get_prompt_lengths_on_ui = Mugen.get_prompt_lengths_on_ui
    original_get_learned_conditioning = Mugen.get_learned_conditioning

    def jina_only_init(self, estimated_config, huggingface_components):
        if "text_encoder" in huggingface_components and "text_encoder_2" in huggingface_components:
            return original_init(self, estimated_config, huggingface_components)

        from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
        from backend.modules.k_prediction import PredictionDiscreteFlow
        from backend.nn.unet import Timestep
        from backend.patcher.clip import CLIP
        from backend.patcher.unet import UnetPatcher
        from backend.patcher.vae import VAE

        missing = [k for k in ("text_encoder", "text_encoder_2") if k not in huggingface_components]
        logger.warning("Initializing Jina-only Mugen without bundled CLIP components: %s", ", ".join(missing))

        ForgeDiffusionEngine.__init__(self, estimated_config, huggingface_components)

        clip = CLIP(model_dict={}, tokenizer_dict={})
        vae = VAE(model=huggingface_components["vae"], is_mugen=True)
        k_predictor = PredictionDiscreteFlow(estimated_config)
        unet = UnetPatcher.from_model(
            model=huggingface_components["unet"],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config,
        )

        self.text_processing_engine_l = None
        self.text_processing_engine_g = None
        self.embedder = Timestep(256)

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_sdxl = True
        self.use_shift = True
        self.jina_missing_clip = True

    def set_clip_skip_jina_safe(self, clip_skip):
        if getattr(self, "text_processing_engine_l", None) is None or getattr(self, "text_processing_engine_g", None) is None:
            return
        return original_set_clip_skip(self, clip_skip)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui_jina_safe(self, prompt):
        if getattr(self, "text_processing_engine_l", None) is None:
            token_count = len(str(prompt).split())
            return token_count, max(77, ((token_count + 76) // 77) * 77)
        return original_get_prompt_lengths_on_ui(self, prompt)

    @torch.inference_mode()
    def get_learned_conditioning_jina_safe(self, prompt):
        if getattr(self, "text_processing_engine_l", None) is None:
            raise RuntimeError(
                "This Mugen checkpoint does not include SDXL CLIP weights. "
                "Enable the 'Jina CLIP v2 SDXL Adapter' extension panel and provide the Jina model + adapter paths."
            )
        return original_get_learned_conditioning(self, prompt)

    Mugen.__init__ = jina_only_init
    Mugen.set_clip_skip = set_clip_skip_jina_safe
    Mugen.get_prompt_lengths_on_ui = get_prompt_lengths_on_ui_jina_safe
    Mugen.get_learned_conditioning = get_learned_conditioning_jina_safe
    Mugen._jina_missing_clip_patch_installed = True

