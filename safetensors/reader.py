"""Zero-copy safetensors reader.

File layout
-----------
The ``.safetensors`` format stores:
1) 8-byte little-endian unsigned 64-bit integer: ``header_len``
2) ``header_len`` bytes of UTF-8 JSON. Example per tensor entry:
   {"tensor_name": {"dtype": "F32", "shape": [2,3], "data_offsets": [start, end]}}
   Offsets are relative to the start of the data section (i.e., byte 8+header_len).
3) Raw tensor data bytes, concatenated in the order implied by offsets.

Why zero-copy?
--------------
We memory-map the file (read-only) and produce NumPy arrays with
``np.frombuffer(mm, dtype, count, offset=abs_start)``. This references the
underlying mapping directly (no copy) and reshapes it to the target shape.

Dependencies: standard library + NumPy.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np

from utils.mmap_ro import ro_mmap


_ITEMSIZE = {
    # floating
    "F16": 2,
    "BF16": 2,  # NumPy 2.0 has bfloat16, otherwise fallback to uint16 view
    "F32": 4,
    "F64": 8,
    # signed ints
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    # unsigned ints
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
    # bool
    "BOOL": 1,
}


def _numpy_dtype(dtype_str: str) -> np.dtype:
    """Map safetensors dtype string to NumPy dtype.

    Returns a NumPy dtype. For BF16, if NumPy lacks bfloat16 support,
    returns ``np.uint16`` as a conservative fallback (still zero-copy view).
    """
    d = dtype_str.upper()
    if d == "F16":
        return np.dtype("float16")
    if d == "BF16":
        # NumPy 2.0+: np.dtype("bfloat16") exists
        try:
            return np.dtype("bfloat16")  # type: ignore[attr-defined]
        except TypeError:
            return np.dtype("uint16")
    if d == "F32":
        return np.dtype("float32")
    if d == "F64":
        return np.dtype("float64")
    if d == "I8":
        return np.dtype("int8")
    if d == "I16":
        return np.dtype("int16")
    if d == "I32":
        return np.dtype("int32")
    if d == "I64":
        return np.dtype("int64")
    if d == "U8":
        return np.dtype("uint8")
    if d == "U16":
        return np.dtype("uint16")
    if d == "U32":
        return np.dtype("uint32")
    if d == "U64":
        return np.dtype("uint64")
    if d == "BOOL":
        return np.dtype("bool")
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _prod(dims: List[int]) -> int:
    n = 1
    for v in dims:
        n *= int(v)
    return int(n)


@dataclass
class _Entry:
    name: str
    dtype: str
    shape: List[int]
    rel_start: int  # relative to data section
    rel_end: int    # relative to data section
    abs_start: int  # absolute in file
    abs_end: int    # absolute in file


class SafeTensorFile:
    """Memory-mapped, zero-copy reader for ``.safetensors`` files.

    Why it exists / When to use:
    - Load model weights lazily, slice-free, and without copies.
    - Efficiently inspect metadata and obtain NumPy views over on-disk data.

    Notes
    -----
    - Keep this object (and its underlying mmap) alive while any returned
      NumPy views or memoryviews are in use.
    - Offsets in the file header are relative to the start of the data block.
    - ``get(..., as_numpy=True)`` returns a non-owning NumPy view using
      ``np.frombuffer(self._mm, dtype, count, offset)``.
    """

    def __init__(self, path: str):
        self._path = path
        self._cm = None  # context manager handle to ro_mmap
        self._mm = None  # type: ignore[assignment]
        self._size = 0
        self._entries: Dict[str, _Entry] = {}

    def __enter__(self) -> "SafeTensorFile":
        self._cm = ro_mmap(self._path)
        self._mm, self._size = self._cm.__enter__()
        self._parse_header()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Close mmap/file via the context manager
        if self._cm is not None:
            return self._cm.__exit__(exc_type, exc, tb)
        return False

    # --- Public API ---
    def keys(self) -> Iterable[str]:
        return self._entries.keys()

    def meta(self, name: str) -> Dict:
        if name not in self._entries:
            raise KeyError(f"No such tensor: {name}")
        e = self._entries[name]
        return {
            "dtype": e.dtype,
            "shape": list(e.shape),
            # Return absolute byte range to make it easy to reason about file layout
            "data_offsets": [e.abs_start, e.abs_end],
        }

    def get(self, name: str, *, as_numpy: bool = True):
        """Get a zero-copy view to the tensor data and its metadata.

        Parameters
        ----------
        name : str
            Tensor name.
        as_numpy : bool, default True
            If True, returns a NumPy array view. Otherwise returns a memoryview.

        Returns
        -------
        (array_like, info_dict)
            ``array_like`` is a NumPy view (when ``as_numpy=True``) or a
            Python ``memoryview``. ``info_dict`` matches :meth:`meta`.
        """
        if name not in self._entries:
            raise KeyError(f"No such tensor: {name}")
        e = self._entries[name]
        info = self.meta(name)

        if as_numpy:
            np_dt = _numpy_dtype(e.dtype)
            count = _prod(e.shape)
            # Zero-copy view into the mmap using an absolute offset
            arr = np.frombuffer(self._mm, dtype=np_dt, count=count, offset=e.abs_start)
            try:
                arr = arr.reshape(e.shape)
            except ValueError as ve:
                raise ValueError(f"Invalid reshape for {name}: {e.shape}") from ve
            return arr, info

        # Raw bytes view over the mapping
        mv = memoryview(self._mm)[e.abs_start:e.abs_end]
        return mv, info

    # --- Internal ---
    def _parse_header(self) -> None:
        if self._size < 8:
            raise ValueError("File too small for safetensors header")

        # First 8 bytes: little-endian u64 header length
        try:
            header_len = struct.unpack_from('<Q', self._mm, 0)[0]
        except struct.error as e:
            raise ValueError(f"Failed to parse header length: {e}") from e

        header_start = 8
        header_end = header_start + int(header_len)
        if header_end > self._size:
            raise ValueError("Declared header length exceeds file size")

        # Decode JSON header
        try:
            header_bytes = bytes(self._mm[header_start:header_end])
            meta = json.loads(header_bytes.decode('utf-8'))
        except Exception as e:
            raise ValueError(f"Invalid header JSON: {e}") from e

        if not isinstance(meta, dict):
            raise ValueError("Header JSON must be a dict of name -> entry")

        data_base = header_end

        for name, entry in meta.items():
            if not isinstance(name, str):
                raise ValueError("Tensor name must be a string")
            if not isinstance(entry, dict):
                raise ValueError(f"Entry for {name} must be an object")

            try:
                dtype = entry["dtype"]
                shape = entry["shape"]
                rel_start, rel_end = entry["data_offsets"]
            except Exception as e:
                raise ValueError(f"Entry for {name} missing fields: {e}") from e

            if dtype not in _ITEMSIZE:
                raise ValueError(f"Unsupported dtype in header for {name}: {dtype}")

            if (not isinstance(shape, list)) or any((not isinstance(x, int) or x < 0) for x in shape):
                raise ValueError(f"Invalid shape for {name}: {shape}")

            if (not isinstance(rel_start, int)) or (not isinstance(rel_end, int)) or rel_start < 0 or rel_end < rel_start:
                raise ValueError(f"Invalid data_offsets for {name}: {entry.get('data_offsets')}")

            itemsize = _ITEMSIZE[dtype]
            expect_nbytes = _prod(shape) * itemsize
            if (rel_end - rel_start) != expect_nbytes:
                raise ValueError(
                    f"Byte size mismatch for {name}: offsets span {(rel_end-rel_start)} bytes, expected {expect_nbytes}"
                )

            abs_start = data_base + rel_start
            abs_end = data_base + rel_end
            if abs_end > self._size:
                raise ValueError(f"Offsets for {name} exceed file size: [{abs_start}, {abs_end}) > {self._size}")

            self._entries[name] = _Entry(
                name=name,
                dtype=dtype,
                shape=list(map(int, shape)),
                rel_start=rel_start,
                rel_end=rel_end,
                abs_start=abs_start,
                abs_end=abs_end,
            )


