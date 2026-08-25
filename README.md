# Jina CLIP v2 SDXL Adapter V2/V3 for Forge Neo
## Fully vibecoded

This Forge extension replaces SDXL/Mugen prompt conditioning with Jina CLIP v2 plus either a V2 or redesigned V3 Jina-to-SDXL adapter.

This extension adds support for jina-clip-v2 adapter as a text encoder for Mugen.
- [jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2)
- [Adapter+UNet](https://huggingface.co/TheRemixer/Mugen-Jina-V2.0)

Place or clone this directory under Forge Neo's `extensions/` directory, then restart Forge.

- Put a local Jina CLIP v2 model under `models/llm/`, `models/LLM/`, `models/Jina/`, or `models/text_encoder/`. A Hugging Face model ID such as `jinaai/jina-clip-v2` can also be entered in the UI.
- Put adapter `.safetensors` files under `models/llm_adapter/`, `models/llm_adapters/`, `models/LLM_Adapters/`, or `models/Jina_Adapters/`.

Open **Jina CLIP v2 SDXL Adapter (V2/V3)** in txt2img or img2img, enable it, and select the model and adapter.

## Adapter versions

The default `auto` architecture selection examines the checkpoint:

- redesigned V3 checkpoints contain `layer_fusion.*` and `mean_pooled_projection.*` tensors;
- V2 checkpoints contain the existing `seq_projection.*` and `attention_blocks.*` sequence path.

V3 captures Jina text layers h8 and h16 plus the final normalized h24 state, performs token-wise layer fusion, and uses the checkpoint's V3 pooled projection. A V3 checkpoint must be complete and shape-compatible; partial or mislabeled V3 checkpoints are rejected instead of being loaded silently.

The V2 max-sequence, attention-pooling, and positional-embedding controls apply only to V2. `Convert legacy fused-MHA adapter` converts older `in_proj_weight` / `in_proj_bias` checkpoints to the current explicit q/k/v layout. Current V2 and V3 checkpoints do not need it.

The optional cross-attention padding mask uses Forge's existing extension patch and works with either adapter version. Prompt weighting, `@ artist` formatting, CLIP-less Mugen checkpoint loading, and the Forge conditioning override remain compatible with V2.

## Notes

- The Jina model must expose its 24-layer text transformer; V3 specifically requires h8, h16, and h24.
- The tokenizer is loaded with `fix_mistral_regex=False`, matching the Jina XLM-R token IDs used during V2/V3 adapter training.
- Loading uses `trust_remote_code=True`, as required by Jina CLIP v2.
- `Local files only` requires all Jina model and trusted-code dependencies to already exist locally.
- Adapter conditioning is emitted as SDXL-compatible 2048-wide cross-attention plus a 1280-wide pooled vector. When the padding mask is enabled, its extra channel is removed by the Forge attention patch before the U-Net projections.
