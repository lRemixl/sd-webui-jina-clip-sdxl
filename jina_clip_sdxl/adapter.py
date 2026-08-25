import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Mapping, Sequence


class AttentionPooler(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        q = self.query.expand(batch_size, -1, -1)
        key_padding_mask = ~mask.bool() if mask is not None else None
        attn_out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return self.norm(attn_out.squeeze(1))


class ExplicitMultiheadAttention(nn.Module):
    """LoRA-friendly Q/K/V attention used by the training fork."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, average_attn_weights=True):
        if need_weights:
            raise ValueError("need_weights=True is not supported by ExplicitMultiheadAttention")

        batch_size = query.shape[0]
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)

        dropout_p = self.dropout.p if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out_proj(attn_output), None


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=16, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ExplicitMultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, mask=None):
        normed = self.norm1(x)
        key_padding_mask = ~mask.bool() if mask is not None else None
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class JinaToSDXLAdapterV2(nn.Module):
    adapter_version = "v2"

    def __init__(
        self,
        llm_dim=1024,
        sdxl_seq_dim=2048,
        sdxl_pooled_dim=1280,
        n_attention_blocks=4,
        num_heads=16,
        dropout=0,
        max_seq_len=539,
        attn_pooling=True,
        use_positional=True,
    ):
        super().__init__()
        self.attn_pooling = attn_pooling
        self.use_positional = use_positional

        self.seq_projection = nn.Sequential(
            nn.Linear(llm_dim, sdxl_seq_dim),
            nn.LayerNorm(sdxl_seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sdxl_seq_dim, sdxl_seq_dim),
        )
        if self.use_positional:
            self.positional_embedding = nn.Embedding(max_seq_len, sdxl_seq_dim)

        self.attention_blocks = nn.ModuleList(
            [
                TransformerBlock(sdxl_seq_dim, num_heads=num_heads, mlp_ratio=4.0, dropout=dropout)
                for _ in range(n_attention_blocks)
            ]
        )

        if self.attn_pooling:
            self.attention_pooler = AttentionPooler(sdxl_seq_dim)
            self.pooled_projection = nn.Linear(sdxl_seq_dim, sdxl_pooled_dim)
        else:
            self.pooled_projection = nn.Sequential(
                nn.Linear(llm_dim, llm_dim),
                nn.LayerNorm(llm_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(llm_dim, sdxl_pooled_dim),
            )

    @property
    def max_seq_len(self):
        if not self.use_positional:
            return None
        return int(self.positional_embedding.weight.shape[0])

    def forward(self, jina_hidden_states, jina_mean_pooled_state, attention_mask=None):
        hidden_states = self.seq_projection(jina_hidden_states)
        if self.use_positional:
            seq_len = hidden_states.size(1)
            if seq_len > self.positional_embedding.weight.shape[0]:
                raise ValueError(
                    f"Jina adapter positional embedding supports {self.positional_embedding.weight.shape[0]} tokens, "
                    f"but got {seq_len}."
                )
            positions = torch.arange(seq_len, device=hidden_states.device)
            hidden_states = hidden_states + self.positional_embedding(positions).unsqueeze(0)

        for block in self.attention_blocks:
            hidden_states = block(hidden_states, attention_mask)

        if self.attn_pooling:
            pooled_features = self.attention_pooler(hidden_states, attention_mask)
            pooled_output = self.pooled_projection(pooled_features)
        else:
            pooled_output = self.pooled_projection(jina_mean_pooled_state)

        return hidden_states, pooled_output


class LayerwiseAttentionFusion(nn.Module):
    """Fuse h8/h16/h24 per token without mixing sequence positions."""

    def __init__(self, dim=1024, num_layers=3, num_blocks=2, num_heads=16, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        if num_layers < 2:
            raise ValueError("LayerwiseAttentionFusion requires at least two layers.")
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.input_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.depth_embeddings = nn.Parameter(torch.zeros(1, num_layers, dim))
        nn.init.normal_(self.depth_embeddings, mean=0.0, std=0.02)
        self.attention_blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(num_blocks)]
        )
        self.layer_score = nn.Linear(dim, 1)
        self.output_projection = nn.Linear(dim, dim)
        self.channel_gate = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states):
        if hidden_states.dim() != 4:
            raise ValueError(
                "LayerwiseAttentionFusion expects [batch, layers, sequence, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )
        batch, layers, sequence, width = hidden_states.shape
        if layers != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} selected layers, got {layers}.")
        if width != self.dim:
            raise ValueError(f"Expected hidden width {self.dim}, got {width}.")

        normalized = torch.stack(
            [self.input_norms[index](hidden_states[:, index]) for index in range(layers)],
            dim=1,
        )
        layer_tokens = normalized.permute(0, 2, 1, 3).reshape(batch * sequence, layers, width)
        layer_tokens = layer_tokens + self.depth_embeddings.to(device=layer_tokens.device, dtype=layer_tokens.dtype)
        for block in self.attention_blocks:
            layer_tokens = block(layer_tokens)

        weights = torch.softmax(self.layer_score(layer_tokens).squeeze(-1), dim=1)
        delta = torch.sum(layer_tokens * weights.unsqueeze(-1), dim=1)
        return self.output_projection(delta).reshape(batch, sequence, width)

    def gated_residual(self, final_hidden_state, delta):
        gate = torch.tanh(self.channel_gate).to(device=delta.device, dtype=delta.dtype)
        return final_hidden_state + delta * gate.view(1, 1, -1)


class JinaToSDXLAdapterV3(nn.Module):
    """Redesigned V3 adapter using token-wise fusion of Jina h8/h16/h24."""

    adapter_version = "v3"
    architecture_revision = 2
    required_hidden_state_layers = (8, 16, 24)
    hidden_state_input_key = "jina_hidden_states_selected_layers"

    def __init__(
        self,
        llm_dim=1024,
        sdxl_seq_dim=2048,
        sdxl_pooled_dim=1280,
        n_attention_blocks=4,
        num_heads=16,
        dropout=0.0,
    ):
        super().__init__()
        self.llm_dim = int(llm_dim)
        self.num_selected_layers = len(self.required_hidden_state_layers)
        self.layer_fusion = LayerwiseAttentionFusion(
            dim=llm_dim,
            num_layers=self.num_selected_layers,
            num_blocks=2,
            num_heads=num_heads,
            mlp_ratio=2.0,
            dropout=dropout,
        )
        self.seq_projection = nn.Sequential(
            nn.Linear(llm_dim, sdxl_seq_dim),
            nn.LayerNorm(sdxl_seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sdxl_seq_dim, sdxl_seq_dim),
        )
        self.attention_blocks = nn.ModuleList(
            [TransformerBlock(sdxl_seq_dim, num_heads=num_heads, mlp_ratio=4.0, dropout=dropout) for _ in range(n_attention_blocks)]
        )
        self.mean_pooled_projection = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, sdxl_pooled_dim),
        )

    @staticmethod
    def _validated_mask(attention_mask, expected_shape: Sequence[int], device):
        if attention_mask is None:
            return None
        if attention_mask.dim() != 2 or tuple(attention_mask.shape) != tuple(expected_shape):
            raise ValueError(
                f"attention_mask must have shape {tuple(expected_shape)}, got {tuple(attention_mask.shape)}."
            )
        return attention_mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _zero_padding(hidden_states, valid_mask):
        return hidden_states if valid_mask is None else hidden_states.masked_fill(~valid_mask.unsqueeze(-1), 0)

    def forward(self, jina_hidden_states_selected_layers, jina_mean_pooled_state, attention_mask=None):
        selected = jina_hidden_states_selected_layers
        if selected.dim() != 4:
            raise ValueError(
                "jina_hidden_states_selected_layers must be [batch, 3, sequence, 1024], "
                f"got {tuple(selected.shape)}."
            )
        if selected.shape[1] != self.num_selected_layers:
            raise ValueError(
                f"Expected h8/h16/h24 ({self.num_selected_layers} layers), got {selected.shape[1]}."
            )
        if selected.shape[-1] != self.llm_dim:
            raise ValueError(f"Expected Jina hidden size {self.llm_dim}, got {selected.shape[-1]}.")
        if jina_mean_pooled_state.dim() != 2:
            raise ValueError(
                f"jina_mean_pooled_state must be [batch, {self.llm_dim}], got {tuple(jina_mean_pooled_state.shape)}."
            )
        if jina_mean_pooled_state.shape[0] != selected.shape[0]:
            raise ValueError("Sequence and pooled Jina states have different batch sizes.")
        if jina_mean_pooled_state.shape[-1] != self.llm_dim:
            raise ValueError(
                f"Expected pooled Jina hidden size {self.llm_dim}, got {jina_mean_pooled_state.shape[-1]}."
            )

        valid_mask = self._validated_mask(
            attention_mask,
            (selected.shape[0], selected.shape[2]),
            selected.device,
        )
        delta = self.layer_fusion(selected)
        hidden_states = self.layer_fusion.gated_residual(selected[:, -1], delta)
        hidden_states = self._zero_padding(hidden_states, valid_mask)
        hidden_states = self._zero_padding(self.seq_projection(hidden_states), valid_mask)
        for block in self.attention_blocks:
            hidden_states = self._zero_padding(block(hidden_states, valid_mask), valid_mask)
        return hidden_states, self.mean_pooled_projection(jina_mean_pooled_state)


def convert_state_dict_for_explicit_attention(old_state_dict):
    new_state_dict = {}
    for key, value in old_state_dict.items():
        if "in_proj_weight" in key:
            q_w, k_w, v_w = value.chunk(3, dim=0)
            base_key = key.replace("in_proj_weight", "")
            new_state_dict[base_key + "q_proj.weight"] = q_w
            new_state_dict[base_key + "k_proj.weight"] = k_w
            new_state_dict[base_key + "v_proj.weight"] = v_w
        elif "in_proj_bias" in key:
            q_b, k_b, v_b = value.chunk(3, dim=0)
            base_key = key.replace("in_proj_bias", "")
            new_state_dict[base_key + "q_proj.bias"] = q_b
            new_state_dict[base_key + "k_proj.bias"] = k_b
            new_state_dict[base_key + "v_proj.bias"] = v_b
        else:
            new_state_dict[key] = value
    return new_state_dict


def detect_adapter_version(state_dict: Mapping[str, torch.Tensor]):
    if any(key.startswith("layer_fusion.") for key in state_dict) or any(
        key.startswith("mean_pooled_projection.") for key in state_dict
    ):
        return "v3"
    if any(key.startswith("seq_projection.") for key in state_dict) and any(
        key.startswith("attention_blocks.") for key in state_dict
    ):
        return "v2"
    raise ValueError("The checkpoint is not a recognized Jina-to-SDXL V2 or redesigned-V3 adapter.")


def missing_or_mismatched_keys(module: nn.Module, state_dict: Mapping[str, torch.Tensor]):
    missing = []
    for key, target in module.state_dict().items():
        value = state_dict.get(key)
        if value is None or tuple(value.shape) != tuple(target.shape):
            missing.append(key)
    return missing


def build_adapter(
    state_dict,
    requested_version="auto",
    v2_max_seq_len=539,
    v2_attn_pooling=True,
    v2_use_positional=True,
):
    """Build an adapter, requiring a complete redesigned V3 checkpoint."""
    detected_version = detect_adapter_version(state_dict)
    requested_version = (requested_version or "auto").lower()
    if requested_version not in ("auto", "v2", "v3"):
        raise ValueError(f"Unknown Jina adapter version selection: {requested_version}")
    if requested_version != "auto" and requested_version != detected_version:
        raise ValueError(
            f"Adapter was selected as {requested_version.upper()}, but its checkpoint is {detected_version.upper()}."
        )

    if detected_version == "v3":
        adapter = JinaToSDXLAdapterV3()
        missing = missing_or_mismatched_keys(adapter, state_dict)
        unexpected = sorted(set(state_dict) - set(adapter.state_dict()))
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing/incompatible: " + ", ".join(missing[:8]))
            if unexpected:
                details.append("unexpected: " + ", ".join(unexpected[:8]))
            raise ValueError("The selected checkpoint is not a complete Jina V3 adapter (" + "; ".join(details) + ")")
        adapter.load_state_dict(state_dict, strict=True)
        return adapter, detected_version, [], []

    adapter = JinaToSDXLAdapterV2(
        max_seq_len=int(v2_max_seq_len),
        attn_pooling=bool(v2_attn_pooling),
        use_positional=bool(v2_use_positional),
    )
    incompatible = adapter.load_state_dict(state_dict, strict=False)
    return adapter, detected_version, list(incompatible.missing_keys), list(incompatible.unexpected_keys)
