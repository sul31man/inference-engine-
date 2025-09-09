"""Minimal safetensors writer for tests/dev.

Why it exists / When to use:
- Generate small, valid ``.safetensors`` files for unit tests and demos.
- Not a full-featured exporter; validates dtype, shapes and produces offsets.
"""

from __future__ import annotations

import io
import json
import struct
from typing import Dict, Tuple

import numpy as np


_ITEMSIZE = {
    "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
    "I8": 1, "I16": 2, "I32": 4, "I64": 8,
    "U8": 1, "U16": 2, "U32": 4, "U64": 8,
    "BOOL": 1,
}

_NP_DTYPE = {
    "F16": np.float16,
    "BF16": getattr(np, 'bfloat16', np.uint16),  # fallback view
    "F32": np.float32,
    "F64": np.float64,
    "I8": np.int8,  "I16": np.int16,  "I32": np.int32,  "I64": np.int64,
    "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "U64": np.uint64,
    "BOOL": np.bool_,
}


def write_min(path: str, tensors: Dict[str, Tuple[np.ndarray, str]]) -> None:
    """Write a minimal ``.safetensors`` file.

    Parameters
    ----------
    path : str
        Output file path.
    tensors : dict
        Mapping: name -> (numpy array, dtype string). The dtype string must
        match the array's dtype when mapped via the internal table.
    """
    # Validate and compute header entries
    entries = {}
    raw_blobs = []
    cursor = 0

    for name, (arr, dts) in tensors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Tensor name must be a non-empty string")
        if dts not in _ITEMSIZE:
            raise ValueError(f"Unsupported dtype: {dts}")

        expect_dt = _NP_DTYPE[dts]
        if arr.dtype != np.dtype(expect_dt):
            raise ValueError(f"Array dtype {arr.dtype} does not match declared {dts}")

        # Ensure C-contiguous raw bytes
        carr = np.ascontiguousarray(arr)
        blob = carr.tobytes(order='C')
        nbytes = len(blob)

        entries[name] = {
            "dtype": dts,
            "shape": list(map(int, carr.shape)),
            "data_offsets": [cursor, cursor + nbytes],
        }
        raw_blobs.append(blob)
        cursor += nbytes

    header_bytes = json.dumps(entries, separators=(',', ':')).encode('utf-8')
    header_len = len(header_bytes)

    with open(path, 'wb') as f:
        # 8-byte LE u64 header length
        f.write(struct.pack('<Q', header_len))
        # header JSON
        f.write(header_bytes)
        # concatenated raw data
        for blob in raw_blobs:
            f.write(blob)


