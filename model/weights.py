from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Iterable

import numpy as np

from safetensors.reader import SafeTensorFile
from model.config import ModelCfg


@dataclass
class ModelWeights:
    """Holds references (views) to per-layer weights.

    This structure stores NumPy views backed by the mmapped safetensors file.
    Ownership remains with the ``SafeTensorFile``; keep it alive while using these.
    """

    # token embedding and final norm/proj (examples, adjust as needed)
    tok_embed: np.ndarray
    final_proj: np.ndarray | None

    # per-layer collections (simple example set for a Transformer block)
    qkv_w: Dict[int, np.ndarray]
    qkv_b: Dict[int, np.ndarray] | None
    out_w: Dict[int, np.ndarray]
    out_b: Dict[int, np.ndarray] | None
    ln1_g: Dict[int, np.ndarray]
    ln1_b: Dict[int, np.ndarray] | None
    ln2_g: Dict[int, np.ndarray]
    ln2_b: Dict[int, np.ndarray] | None
    ffn_w1: Dict[int, np.ndarray]
    ffn_w2: Dict[int, np.ndarray]


def _shape_str(a: np.ndarray) -> str:
    return "x".join(str(x) for x in a.shape)


def bind_safetensors(st: SafeTensorFile, cfg: ModelCfg) -> ModelWeights:
    """Bind safetensors entries to a ``ModelWeights`` instance.

    Expected naming (example):
      - "tok_embed.weight" -> (vocab_size, d_model)
      - per-layer i in [0..n_layers-1]:
        - f"layers.{i}.attn.qkv.weight" -> (d_model, 3*d_model)
        - f"layers.{i}.attn.out.weight" -> (d_model, d_model)
        - f"layers.{i}.ln1.g" -> (d_model,)  (gamma)
        - f"layers.{i}.ln2.g" -> (d_model,)
        - f"layers.{i}.ffn.w1.weight" -> (d_model, 4*d_model)
        - f"layers.{i}.ffn.w2.weight" -> (4*d_model, d_model)
    Adjust to your actual model's naming.
    """
    expect_tok = (cfg.vocab_size, cfg.d_model)

    def get_arr(name: str) -> np.ndarray:
        arr, info = st.get(name)
        return arr

    # mandatory
    tok_embed = get_arr("tok_embed.weight")
    if tuple(tok_embed.shape) != expect_tok:
        raise ValueError(f"tok_embed expected {expect_tok}, got {tuple(tok_embed.shape)}")

    qkv_w: Dict[int, np.ndarray] = {}
    out_w: Dict[int, np.ndarray] = {}
    ln1_g: Dict[int, np.ndarray] = {}
    ln2_g: Dict[int, np.ndarray] = {}
    ffn_w1: Dict[int, np.ndarray] = {}
    ffn_w2: Dict[int, np.ndarray] = {}

    for i in range(cfg.n_layers):
        qkv = get_arr(f"layers.{i}.attn.qkv.weight")
        out = get_arr(f"layers.{i}.attn.out.weight")
        g1 = get_arr(f"layers.{i}.ln1.g")
        g2 = get_arr(f"layers.{i}.ln2.g")
        w1 = get_arr(f"layers.{i}.ffn.w1.weight")
        w2 = get_arr(f"layers.{i}.ffn.w2.weight")

        if tuple(qkv.shape) != (cfg.d_model, 3 * cfg.d_model):
            raise ValueError(f"layers.{i}.attn.qkv.weight bad shape: {tuple(qkv.shape)}")
        if tuple(out.shape) != (cfg.d_model, cfg.d_model):
            raise ValueError(f"layers.{i}.attn.out.weight bad shape: {tuple(out.shape)}")
        if tuple(g1.shape) != (cfg.d_model,):
            raise ValueError(f"layers.{i}.ln1.g bad shape: {tuple(g1.shape)}")
        if tuple(g2.shape) != (cfg.d_model,):
            raise ValueError(f"layers.{i}.ln2.g bad shape: {tuple(g2.shape)}")
        if tuple(w1.shape) != (cfg.d_model, 4 * cfg.d_model):
            raise ValueError(f"layers.{i}.ffn.w1.weight bad shape: {tuple(w1.shape)}")
        if tuple(w2.shape) != (4 * cfg.d_model, cfg.d_model):
            raise ValueError(f"layers.{i}.ffn.w2.weight bad shape: {tuple(w2.shape)}")

        qkv_w[i] = qkv
        out_w[i] = out
        ln1_g[i] = g1
        ln2_g[i] = g2
        ffn_w1[i] = w1
        ffn_w2[i] = w2

    return ModelWeights(
        tok_embed=tok_embed,
        final_proj=None,
        qkv_w=qkv_w,
        qkv_b=None,
        out_w=out_w,
        out_b=None,
        ln1_g=ln1_g,
        ln1_b=None,
        ln2_g=ln2_g,
        ln2_b=None,
        ffn_w1=ffn_w1,
        ffn_w2=ffn_w2,
    )


