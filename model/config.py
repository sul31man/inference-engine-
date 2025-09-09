from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ModelCfg:
    """Minimal model configuration.

    Why it exists / When to use:
    - Captures architectural hyperparameters used to validate weight shapes
      and to build the compute graph later.
    """

    # core dims
    d_model: int
    n_layers: int
    n_heads: int
    vocab_size: int

    # rotary/rope parameters (simple form)
    rope_theta: float = 10000.0
    rope_dim: int = 0  # 0 => infer from d_model/n_heads if needed


def load_cfg(path: str) -> ModelCfg:
    """Load model configuration from a JSON file.

    The JSON should contain at least: ``d_model, n_layers, n_heads, vocab_size``.
    Optional: ``rope_theta, rope_dim``.
    """
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    required = ["d_model", "n_layers", "n_heads", "vocab_size"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing required config field: {k}")
    return ModelCfg(
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        vocab_size=int(cfg["vocab_size"]),
        rope_theta=float(cfg.get("rope_theta", 10000.0)),
        rope_dim=int(cfg.get("rope_dim", 0)),
    )


