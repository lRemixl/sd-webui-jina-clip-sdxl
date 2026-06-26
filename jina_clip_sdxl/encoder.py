import gc
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from backend import memory_management
from modules import prompt_parser
from modules.shared import opts

from .adapter import JinaToSDXLAdapterV2, convert_state_dict_for_explicit_attention

logger = logging.getLogger("JinaCLIP-SDXL")


def ensure_flash_attn_bypass():
    try:
        from flash_attn.ops.triton.rotary import apply_rotary  # noqa: F401
        return
    except Exception:
        pass

    for key in [k for k in sys.modules if k.startswith("flash_attn")]:
        del sys.modules[key]


def dtype_from_name(name, device=None):
    name = (name or "auto").lower()
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if device is None:
        device = memory_management.text_encoder_device()
    if torch.device(device).type == "cpu":
        return torch.float32
    dtype = memory_management.text_encoder_dtype(device)
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return dtype
    return torch.bfloat16


def device_from_name(name):
    name = (name or "auto").lower()
    if name == "auto":
        return memory_management.text_encoder_device()
    return torch.device(name)


def _from_pretrained(cls, model_id, local_files_only):
    kwargs = dict(
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    try:
        return cls.from_pretrained(model_id, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return cls.from_pretrained(model_id, **kwargs)


class JinaStates:
    def __init__(self, model_id, device, dtype, max_length=512, local_files_only=False):
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_length = int(max_length)

        logger.info("Loading Jina tokenizer from %s", model_id)
        self.tokenizer = _from_pretrained(AutoTokenizer, model_id, local_files_only)

        logger.info("Loading Jina CLIP v2 from %s", model_id)
        ensure_flash_attn_bypass()
        self.model = AutoModel.from_pretrained(
            model_id,
            low_cpu_mem_usage=False,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)

        if hasattr(self.model, "vision_model"):
            del self.model.vision_model
            memory_management.soft_empty_cache()

        self.model.eval()
        self.model.requires_grad_(False)

        self.hidden_states_cache = None
        self.encoder_module = self._find_text_encoder_module()
        self.encoder_module.register_forward_hook(self._forward_hook)
        logger.info("Attached Jina hidden-state hook to %s", self.encoder_module.__class__.__name__)

    def _find_text_encoder_module(self):
        for name, module in self.model.named_modules():
            if "vision" in name.lower():
                continue

            for attr in ("layer", "layers", "block", "blocks"):
                layer_list = getattr(module, attr, None)
                if isinstance(layer_list, torch.nn.ModuleList) and len(layer_list) > 1:
                    return module

        raise RuntimeError("Could not identify Jina CLIP v2 text encoder module for hidden-state hook.")

    def _forward_hook(self, module, args, output):
        if hasattr(output, "last_hidden_state"):
            self.hidden_states_cache = output.last_hidden_state
        elif isinstance(output, tuple):
            self.hidden_states_cache = output[0]
        else:
            self.hidden_states_cache = output

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        self.model.to(device=self.device, dtype=self.dtype)
        return self

    def mean_pooling(self, hidden_states, attention_mask):
        hidden_states_f32 = hidden_states.to(torch.float32)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states_f32.size()).float()
        sum_embeddings = torch.sum(hidden_states_f32 * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return (sum_embeddings / sum_mask).to(self.dtype)

    @torch.inference_mode()
    def run(self, input_ids, attention_mask, output_dtype=torch.float32):
        self.hidden_states_cache = None

        if hasattr(self.model, "get_text_features"):
            pooled = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(out, "text_embeds"):
                pooled = out.text_embeds
            elif hasattr(out, "pooler_output"):
                pooled = out.pooler_output
            elif isinstance(out, tuple):
                pooled = out[1] if len(out) > 1 else out[0]
            else:
                pooled = out

        if self.hidden_states_cache is None:
            raise RuntimeError("Jina hidden-state hook did not capture sequence states.")

        if not isinstance(pooled, torch.Tensor):
            pooled = self.mean_pooling(self.hidden_states_cache, attention_mask)

        return self.hidden_states_cache.clone().to(output_dtype), pooled.clone().to(output_dtype)


def format_artist_tags(text):
    pattern = r"(^|,|\n)\s*(@[^,\.\n]+)"
    extracted_tags = []

    def extract_tag(match):
        prefix = match.group(1)
        tag_content = re.sub(r"\s+", " ", match.group(2).strip())
        extracted_tags.append(re.sub(r"^@\s*", "@ ", tag_content))
        return "\n" if prefix == "\n" else ""

    clean_string = re.sub(pattern, extract_tag, text)
    clean_string = re.sub(r"(,\s*){2,}", ", ", clean_string)
    clean_string = re.sub(r"(^|\n)[ \t,]+", r"\1", clean_string)
    clean_string = re.sub(r"[ \t,]+($|\n)", r"\1", clean_string)

    if not extracted_tags:
        return text
    prefix = ", ".join(extracted_tags)
    return prefix + (", " + clean_string if clean_string else "")


def parse_weighted_text(text):
    clean_parts = []
    char_weights = []

    for segment, weight in prompt_parser.parse_prompt_attention(text):
        if segment == "BREAK" and weight == -1:
            segment = "\n"
            weight = 1.0
        clean_parts.append(segment)
        char_weights.extend([float(weight)] * len(segment))

    return "".join(clean_parts), char_weights


def get_token_data(tokenizer, text, char_weights, device, padding_mode, max_length):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    input_ids = inputs.input_ids[0].to(device)
    attention_mask = inputs.attention_mask[0].to(device)
    offset_mapping = inputs.offset_mapping[0]
    seq_len = int(input_ids.shape[0])

    token_weights = torch.ones(seq_len, dtype=torch.float32, device=device)
    for token_index in range(seq_len):
        start = int(offset_mapping[token_index][0].item())
        end = int(offset_mapping[token_index][1].item())
        if start == end:
            continue
        if start < len(char_weights) and end <= len(char_weights):
            segment_weights = char_weights[start:end]
            if segment_weights:
                token_weights[token_index] = sum(segment_weights) / len(segment_weights)

    info = f"Original Jina token length: {seq_len}"
    target_len = seq_len
    if padding_mode != "none":
        if padding_mode == "Nearest-77":
            target_len = int(math.ceil(seq_len / 77) * 77)
            info += f", padded to nearest multiple of 77: {target_len}"
        else:
            target_len = int(padding_mode)
            info += f", padded to {target_len}"

    if target_len > seq_len:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id or 0
        pad_len = target_len - seq_len
        input_ids = torch.cat(
            [
                input_ids,
                torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype, device=device),
            ]
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.zeros((pad_len,), dtype=attention_mask.dtype, device=device),
            ]
        )
        token_weights = torch.cat(
            [
                token_weights,
                torch.ones((pad_len,), dtype=token_weights.dtype, device=device),
            ]
        )

    return input_ids.unsqueeze(0), attention_mask.unsqueeze(0), token_weights, info


@dataclass(frozen=True)
class JinaConfig:
    model_id: str
    adapter_path: str
    device_name: str = "auto"
    dtype_name: str = "auto"
    max_length: int = 512
    padding_mode: str = "Nearest-77"
    weighting_mode: str = "comfy"
    adapter_seq_len: int = 539
    attn_pooling: bool = True
    use_positional: bool = True
    format_tags: bool = True
    cross_attention_mask: bool = False
    local_files_only: bool = False
    convert_legacy_mha: bool = False


class JinaConditioningManager:
    def __init__(self):
        self.jina = None
        self.jina_key = None
        self.adapter = None
        self.adapter_key = None

    def unload(self):
        self.jina = None
        self.jina_key = None
        self.adapter = None
        self.adapter_key = None
        gc.collect()
        memory_management.soft_empty_cache()

    def _load_jina(self, cfg):
        device = device_from_name(cfg.device_name)
        dtype = dtype_from_name(cfg.dtype_name, device)
        model_id = os.path.expandvars(os.path.expanduser(cfg.model_id))
        key = (model_id, str(device), str(dtype), int(cfg.max_length), bool(cfg.local_files_only))

        if self.jina is None or self.jina_key != key:
            self.jina = None
            gc.collect()
            memory_management.soft_empty_cache()
            self.jina = JinaStates(
                model_id=model_id,
                device=device,
                dtype=dtype,
                max_length=cfg.max_length,
                local_files_only=cfg.local_files_only,
            )
            self.jina_key = key
        else:
            self.jina.to(device=device, dtype=dtype)

        return self.jina

    def _load_adapter(self, cfg):
        adapter_path = str(Path(os.path.expandvars(os.path.expanduser(cfg.adapter_path))))
        device = device_from_name(cfg.device_name)
        key = (
            adapter_path,
            str(device),
            int(cfg.adapter_seq_len),
            bool(cfg.attn_pooling),
            bool(cfg.use_positional),
            bool(cfg.convert_legacy_mha),
        )

        if self.adapter is not None and self.adapter_key == key:
            self.adapter.to(device)
            return self.adapter

        logger.info("Loading Jina SDXL adapter from %s", adapter_path)
        adapter = JinaToSDXLAdapterV2(
            llm_dim=1024,
            sdxl_seq_dim=2048,
            sdxl_pooled_dim=1280,
            n_attention_blocks=4,
            num_heads=16,
            dropout=0,
            max_seq_len=int(cfg.adapter_seq_len),
            attn_pooling=bool(cfg.attn_pooling),
            use_positional=bool(cfg.use_positional),
        )
        checkpoint = load_file(adapter_path, device="cpu")
        if cfg.convert_legacy_mha:
            checkpoint = convert_state_dict_for_explicit_attention(checkpoint)
        missing, unexpected = adapter.load_state_dict(checkpoint, strict=False)
        if missing:
            logger.warning("Jina adapter missing %d keys; first few: %s", len(missing), list(missing)[:8])
        if unexpected:
            logger.warning("Jina adapter ignored %d unexpected keys; first few: %s", len(unexpected), list(unexpected)[:8])
        adapter.to(device)
        adapter.eval()

        self.adapter = adapter
        self.adapter_key = key
        return self.adapter

    @torch.inference_mode()
    def encode_text(self, text, cfg):
        if cfg.format_tags:
            text = format_artist_tags(text)

        jina = self._load_jina(cfg)
        adapter = self._load_adapter(cfg)
        device = jina.device

        clean_text, char_weights = parse_weighted_text(text)
        input_ids, attention_mask, token_weights, info = get_token_data(
            jina.tokenizer,
            clean_text,
            char_weights,
            device,
            cfg.padding_mode,
            cfg.max_length,
        )

        adapter_max = getattr(adapter, "max_seq_len", None)
        if adapter_max is not None and input_ids.shape[1] > adapter_max:
            raise ValueError(
                f"Jina prompt has {input_ids.shape[1]} tokens but adapter max_seq_len is {adapter_max}. "
                "Use a larger adapter sequence length or a shorter prompt."
            )

        output_dtype = torch.float32

        def run(ids, mask):
            hidden, pooled = jina.run(ids, mask, output_dtype=output_dtype)
            return adapter(hidden.to(torch.float32), pooled.to(torch.float32), mask.to(torch.float32))

        prompt_embeds, pooled = run(input_ids, attention_mask)

        if cfg.weighting_mode == "A1111":
            prompt_embeds = prompt_embeds * token_weights.unsqueeze(0).unsqueeze(-1).to(prompt_embeds.device)
        elif cfg.weighting_mode == "comfy" and char_weights:
            empty_inputs = jina.tokenizer("", return_tensors="pt")
            empty_ids = empty_inputs.input_ids.to(device)
            empty_mask = empty_inputs.attention_mask.to(device)
            seq_len = input_ids.shape[1]
            pad_len = seq_len - empty_ids.shape[1]
            if pad_len > 0:
                pad_token_id = jina.tokenizer.pad_token_id or jina.tokenizer.eos_token_id or 0
                empty_ids = torch.cat(
                    [
                        empty_ids,
                        torch.full((1, pad_len), pad_token_id, dtype=empty_ids.dtype, device=device),
                    ],
                    dim=1,
                )
                empty_mask = torch.cat(
                    [
                        empty_mask,
                        torch.zeros((1, pad_len), dtype=empty_mask.dtype, device=device),
                    ],
                    dim=1,
                )
            elif pad_len < 0:
                empty_ids = empty_ids[:, :seq_len]
                empty_mask = empty_mask[:, :seq_len]

            empty_embeds, _ = run(empty_ids, empty_mask)
            weights = token_weights.unsqueeze(0).unsqueeze(-1).to(prompt_embeds.device)
            prompt_embeds = empty_embeds + (prompt_embeds - empty_embeds) * weights

        prompt_embeds = prompt_embeds.cpu().contiguous()
        pooled = pooled.cpu().contiguous()
        attention_mask = attention_mask.cpu().contiguous()
        return prompt_embeds, pooled, attention_mask, info

    @torch.inference_mode()
    def get_conditioning(self, engine, prompts, cfg):
        prompt_embeds_list = []
        pooled_list = []
        masks = []
        infos = []

        for text in prompts:
            prompt_embeds, pooled, attention_mask, info = self.encode_text(str(text), cfg)
            prompt_embeds_list.append(prompt_embeds[0])
            pooled_list.append(pooled[0])
            masks.append(attention_mask[0])
            infos.append(info)

        max_len = max(t.shape[0] for t in prompt_embeds_list)
        for index, tensor in enumerate(prompt_embeds_list):
            if tensor.shape[0] < max_len:
                pad = tensor.new_zeros(max_len - tensor.shape[0], tensor.shape[1])
                prompt_embeds_list[index] = torch.cat([tensor, pad], dim=0)
        for index, tensor in enumerate(masks):
            if tensor.shape[0] < max_len:
                masks[index] = torch.cat([tensor, tensor.new_zeros(max_len - tensor.shape[0])], dim=0)

        crossattn = torch.stack(prompt_embeds_list, dim=0)
        pooled = torch.stack(pooled_list, dim=0)
        attention_mask = torch.stack(masks, dim=0)

        width = getattr(prompts, "width", 1024) or 1024
        height = getattr(prompts, "height", 1024) or 1024
        crop_w = opts.sdxl_crop_left
        crop_h = opts.sdxl_crop_top
        target_width = width
        target_height = height

        out = [
            engine.embedder(torch.Tensor([height])),
            engine.embedder(torch.Tensor([width])),
            engine.embedder(torch.Tensor([crop_h])),
            engine.embedder(torch.Tensor([crop_w])),
            engine.embedder(torch.Tensor([target_height])),
            engine.embedder(torch.Tensor([target_width])),
        ]
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(pooled.shape[0], 1).to(pooled)

        if opts.sdxl_zero_neg and getattr(prompts, "is_negative_prompt", False) and all(x == "" for x in prompts):
            pooled = torch.zeros_like(pooled)
            crossattn = torch.zeros_like(crossattn)

        if cfg.cross_attention_mask:
            crossattn = torch.cat([crossattn, attention_mask.to(dtype=crossattn.dtype).unsqueeze(-1)], dim=-1)

        cond = dict(
            crossattn=crossattn,
            vector=torch.cat([pooled, flat], dim=1),
        )

        engine.extra_generation_params["Jina CLIP v2"] = "enabled"
        engine.extra_generation_params["Jina adapter"] = os.path.basename(cfg.adapter_path)
        engine.extra_generation_params["Jina tokens"] = "; ".join(infos[:2])
        return cond


manager = JinaConditioningManager()
