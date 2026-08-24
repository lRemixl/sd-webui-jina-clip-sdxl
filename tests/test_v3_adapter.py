import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from jina_clip_sdxl import adapter as adapter_module
from jina_clip_sdxl.adapter import (
    JinaToSDXLAdapterV3,
    build_adapter,
    convert_state_dict_for_explicit_attention,
    detect_adapter_version,
)


ROOT = Path(__file__).resolve().parents[1]


def _tiny_v3():
    return JinaToSDXLAdapterV3(
        llm_dim=8,
        sdxl_seq_dim=16,
        sdxl_pooled_dim=6,
        n_attention_blocks=1,
        num_heads=2,
    )


def test_v3_forward_has_sdxl_shapes_and_zeroes_padding():
    model = _tiny_v3().eval()
    selected = torch.randn(2, 3, 5, 8)
    pooled = torch.randn(2, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])

    with torch.no_grad():
        sequence, pooled_output = model(selected, pooled, mask)

    assert sequence.shape == (2, 5, 16)
    assert pooled_output.shape == (2, 6)
    torch.testing.assert_close(sequence[0, 3:], torch.zeros_like(sequence[0, 3:]))
    torch.testing.assert_close(sequence[1, 4:], torch.zeros_like(sequence[1, 4:]))


def test_adapter_version_detection_distinguishes_redesigned_v3():
    assert detect_adapter_version({"layer_fusion.channel_gate": torch.zeros(8)}) == "v3"
    assert detect_adapter_version({"mean_pooled_projection.0.weight": torch.zeros(8, 8)}) == "v3"
    assert detect_adapter_version(
        {
            "seq_projection.0.weight": torch.zeros(8, 8),
            "attention_blocks.0.norm1.weight": torch.zeros(8),
        }
    ) == "v2"


def test_v3_loader_requires_a_complete_shape_compatible_checkpoint(monkeypatch):
    original_class = adapter_module.JinaToSDXLAdapterV3
    checkpoint = _tiny_v3().state_dict()
    monkeypatch.setattr(adapter_module, "JinaToSDXLAdapterV3", _tiny_v3)

    loaded, version, missing, unexpected = build_adapter(checkpoint, "auto")
    assert isinstance(loaded, original_class)
    assert version == "v3"
    assert missing == []
    assert unexpected == []

    incomplete = dict(checkpoint)
    incomplete.pop("layer_fusion.channel_gate")
    with pytest.raises(ValueError, match="complete Jina V3 adapter"):
        build_adapter(incomplete, "v3")

    wrong_shape = dict(checkpoint)
    wrong_shape["layer_fusion.channel_gate"] = torch.zeros(7)
    with pytest.raises(ValueError, match="complete Jina V3 adapter"):
        build_adapter(wrong_shape, "auto")


def test_requested_adapter_version_rejects_a_mislabeled_checkpoint():
    with pytest.raises(ValueError, match="selected as V2.*checkpoint is V3"):
        build_adapter({"layer_fusion.channel_gate": torch.zeros(8)}, "v2")


def test_legacy_fused_attention_conversion_preserves_other_tensors():
    checkpoint = {
        "block.attn.in_proj_weight": torch.arange(48, dtype=torch.float32).reshape(12, 4),
        "block.attn.in_proj_bias": torch.arange(12, dtype=torch.float32),
        "block.attn.out_proj.weight": torch.ones(4, 4),
    }
    converted = convert_state_dict_for_explicit_attention(checkpoint)

    assert "block.attn.in_proj_weight" not in converted
    assert "block.attn.in_proj_bias" not in converted
    assert converted["block.attn.q_proj.weight"].shape == (4, 4)
    assert converted["block.attn.k_proj.bias"].shape == (4,)
    assert converted["block.attn.out_proj.weight"] is checkpoint["block.attn.out_proj.weight"]


def _load_encoder_state_contract():
    path = ROOT / "jina_clip_sdxl" / "encoder.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in {"_sequence_tensor", "_restore_sequence"})
        or (isinstance(node, ast.ClassDef) and node.name == "JinaStates")
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "JinaToSDXLAdapterV3": SimpleNamespace(required_hidden_state_layers=(8, 16, 24)),
        "torch": torch,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["JinaStates"]


def test_jina_state_capture_replaces_raw_h24_hook_with_final_normalized_state():
    states_class = _load_encoder_state_contract()
    states = states_class.__new__(states_class)
    states.dtype = torch.float32
    states.selected_layers = (8, 16, 24)
    states.hidden_states_cache = None
    states.selected_state_cache = [None, None, None]

    final = torch.full((1, 4, 8), 9.0)
    h8 = torch.full((1, 4, 8), 1.0)
    h16 = torch.full((1, 4, 8), 2.0)
    raw_h24 = torch.full((1, 4, 8), 3.0)
    pooled = torch.full((1, 8), 4.0)

    class FakeJina:
        def get_text_features(self, input_ids, attention_mask):
            states.hidden_states_cache = final
            states.selected_state_cache = [h8, h16, raw_h24]
            return pooled

    states.model = FakeJina()
    input_ids = torch.ones(1, 4, dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 0]])
    final_output, selected, pooled_output = states.run(input_ids, mask)

    torch.testing.assert_close(final_output, final)
    torch.testing.assert_close(selected[:, 0], h8)
    torch.testing.assert_close(selected[:, 1], h16)
    torch.testing.assert_close(selected[:, 2], final)
    torch.testing.assert_close(pooled_output, pooled)
    assert states.hidden_states_cache is None
    assert states.selected_state_cache == [None, None, None]


def test_forge_extension_wires_auto_v2_v3_selection_and_selected_states():
    encoder = (ROOT / "jina_clip_sdxl" / "encoder.py").read_text(encoding="utf-8-sig")
    script = (ROOT / "scripts" / "jina_clip_sdxl.py").read_text(encoding="utf-8-sig")
    assert 'adapter_version: str = "auto"' in encoder
    assert 'selected if adapter.adapter_version == "v3" else final_state' in encoder
    assert "fix_mistral_regex=False" in encoder
    assert "validate_jina_tokenizer(self.tokenizer)" in encoder
    assert 'choices=["auto", "v2", "v3"]' in script
    assert 'value="Nearest-77"' in script
    assert "install_global_patches()" in script
    assert "install_unet_mask_wrapper" in script
