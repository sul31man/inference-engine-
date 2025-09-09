import json
import numpy as np
import pytest

from model.config import load_cfg, ModelCfg
from model.weights import bind_safetensors
from safetensors.write_min import write_min
from safetensors.reader import SafeTensorFile


def _make_min_weights(cfg: ModelCfg):
    tensors = {}
    tensors["tok_embed.weight"] = (np.arange(cfg.vocab_size * cfg.d_model, dtype=np.float32).reshape(cfg.vocab_size, cfg.d_model), 'F32')
    for i in range(cfg.n_layers):
        tensors[f"layers.{i}.attn.qkv.weight"] = (np.zeros((cfg.d_model, 3*cfg.d_model), dtype=np.float32), 'F32')
        tensors[f"layers.{i}.attn.out.weight"] = (np.zeros((cfg.d_model, cfg.d_model), dtype=np.float32), 'F32')
        tensors[f"layers.{i}.ln1.g"] = (np.ones((cfg.d_model,), dtype=np.float32), 'F32')
        tensors[f"layers.{i}.ln2.g"] = (np.ones((cfg.d_model,), dtype=np.float32), 'F32')
        tensors[f"layers.{i}.ffn.w1.weight"] = (np.zeros((cfg.d_model, 4*cfg.d_model), dtype=np.float32), 'F32')
        tensors[f"layers.{i}.ffn.w2.weight"] = (np.zeros((4*cfg.d_model, cfg.d_model), dtype=np.float32), 'F32')
    return tensors


def test_cfg_load_and_bind(tmp_path):
    cfg_path = tmp_path / 'model.json'
    cfg_dict = {"d_model": 8, "n_layers": 2, "n_heads": 2, "vocab_size": 16}
    cfg_path.write_text(json.dumps(cfg_dict))
    cfg = load_cfg(str(cfg_path))

    tensors = _make_min_weights(cfg)
    st_path = tmp_path / 'w.safetensors'
    write_min(str(st_path), tensors)

    with SafeTensorFile(str(st_path)) as st:
        W = bind_safetensors(st, cfg)
        assert W.tok_embed.shape == (cfg.vocab_size, cfg.d_model)
        for i in range(cfg.n_layers):
            assert W.qkv_w[i].shape == (cfg.d_model, 3*cfg.d_model)
            assert W.out_w[i].shape == (cfg.d_model, cfg.d_model)
            assert W.ln1_g[i].shape == (cfg.d_model,)
            assert W.ln2_g[i].shape == (cfg.d_model,)
            assert W.ffn_w1[i].shape == (cfg.d_model, 4*cfg.d_model)
            assert W.ffn_w2[i].shape == (4*cfg.d_model, cfg.d_model)


