import torch
import torch.nn as nn
import torch.nn.functional as F


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

