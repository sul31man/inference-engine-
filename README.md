Zero-copy safetensors utilities (Python)
=======================================

What is mmap?
-------------
``mmap`` memory-maps a file so its bytes appear in your virtual memory. The OS pages data lazily on first access. When combined with NumPy's ``frombuffer``, you can create array views over file-backed memory without copying, so large models can be inspected and streamed efficiently.

Safetensors layout (simplified)
-------------------------------
```
[ 8 bytes little-endian u64 header_len ]
[ header_len bytes of JSON metadata ]
[ raw tensor data bytes concatenated ]
```
Each tensor entry in the JSON is: ``{"name": {"dtype": "F32", "shape": [..], "data_offsets": [start, end]}}``.
Offsets are relative to the beginning of the data section (immediately after the JSON).

Zero-copy vs copy
-----------------
NumPy ``frombuffer`` on the ``mmap`` creates a view referencing the mapping. No data is copied. If you slice or call ``np.array(view)`` you will copy. Keep the mapping alive while any views exist.

Example usage
-------------
```python
from safetensors.reader import SafeTensorFile
with SafeTensorFile("model.safetensors") as st:
    W, info = st.get("model.embed.weight")
    print(W.shape, W.dtype, info["data_offsets"])  # zero-copy view
```

Integration note for MLX/Metal
------------------------------
- Keep the file mmapped on host for metadata parsing.
- Upload weights once into an ``MTLBuffer`` for runtime performance.
- To compute offsets for slices (e.g., KV cache or sharded weights), use: ``abs = start + slot * (entry_nbytes / num_slots)``.

How to run
----------
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip numpy pytest
pytest -q
python examples/inspect_and_bench.py /path/to/file.safetensors
```

# inference-engine-
Building my own inference engine from scratch. 
