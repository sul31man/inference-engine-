#!/usr/bin/env python3
"""Download sshleifer/tiny-gpt2 and export weights to simple .bin files.

This script requires: pip install transformers torch safetensors numpy

Outputs under models/tiny_gpt2_bins/:
- model.json (d_model, n_layers, n_heads, vocab_size)
- tok_emb.bin
- lm_head.bin
- Per layer L: Wq_L.bin, Wk_L.bin, Wv_L.bin, Wo_L.bin, W1_L.bin, W3_L.bin, W2_L.bin

Note: GPT-2 MLP uses two projections (c_fc, c_proj). We write W1=c_fc,
W3=c_fc (duplicate to satisfy gated MLP API), W2=c_proj.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig


def save_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype(np.float32, copy=False).tofile(path)


def main():
    out_dir = Path("models/tiny_gpt2_bins")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = "sshleifer/tiny-gpt2"
    cfg = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    sd = model.state_dict()

    d_model = cfg.n_embd
    n_layer = cfg.n_layer
    n_head = cfg.n_head
    vocab_size = cfg.vocab_size
    d_head = d_model // n_head
    d_ff = 4 * d_model

    # Save model.json
    with open(out_dir / "model.json", "w") as f:
        json.dump({
            "d_model": d_model,
            "n_layers": n_layer,
            "n_heads": n_head,
            "vocab_size": vocab_size,
            "rope_theta": 10000.0,
            "rope_dim": 0
        }, f)

    # Token embeddings and LM head
    wte = sd["transformer.wte.weight"].cpu().numpy()  # [vocab, d_model]
    lm_head = sd["lm_head.weight"].cpu().numpy()      # [vocab, d_model]
    save_bin(out_dir / "tok_emb.bin", wte)
    save_bin(out_dir / "lm_head.bin", lm_head)

    # Layers
    for L in range(n_layer):
        prefix = f"transformer.h.{L}."
        # Fused qkv: [d_model, 3*d_model]
        c_attn_w = sd[prefix + "attn.c_attn.weight"].cpu().numpy()
        # Split last dimension into q,k,v
        Wq, Wk, Wv = np.split(c_attn_w, 3, axis=1)  # [d_model, d_model] each
        Wo = sd[prefix + "attn.c_proj.weight"].cpu().numpy()  # [d_model, d_model]

        # MLP
        W1 = sd[prefix + "mlp.c_fc.weight"].cpu().numpy()     # [d_model, 4*d_model]
        W2 = sd[prefix + "mlp.c_proj.weight"].cpu().numpy()   # [4*d_model, d_model]
        W3 = W1  # duplicate to satisfy gated API

        save_bin(out_dir / f"Wq_{L}.bin", Wq)
        save_bin(out_dir / f"Wk_{L}.bin", Wk)
        save_bin(out_dir / f"Wv_{L}.bin", Wv)
        save_bin(out_dir / f"Wo_{L}.bin", Wo)
        save_bin(out_dir / f"W1_{L}.bin", W1)
        save_bin(out_dir / f"W3_{L}.bin", W3)
        save_bin(out_dir / f"W2_{L}.bin", W2)

    print(f"Exported tiny-gpt2 weights to {out_dir}")


if __name__ == "__main__":
    main()


