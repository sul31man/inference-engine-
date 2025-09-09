import os
import json
import struct
import numpy as np
import pytest

from safetensors.reader import SafeTensorFile
from safetensors.write_min import write_min


def test_roundtrip_basic(tmp_path):
    path = tmp_path / 'tensors.safetensors'
    a = (np.arange(16, dtype=np.float16).reshape(4, 4), 'F16')
    b = (np.arange(30, dtype=np.int32).reshape(2, 3, 5), 'I32')
    write_min(str(path), { 'A': a, 'B': b })

    with SafeTensorFile(str(path)) as st:
        keys = sorted(list(st.keys()))
        assert keys == ['A', 'B']

        A, infoA = st.get('A')
        assert A.shape == (4, 4)
        assert A.dtype == np.float16
        assert infoA['dtype'] == 'F16'
        np.testing.assert_array_equal(A, a[0])

        B, infoB = st.get('B')
        assert B.shape == (2, 3, 5)
        assert B.dtype == np.int32
        assert infoB['dtype'] == 'I32'
        np.testing.assert_array_equal(B, b[0])


def test_zero_copy_numpy_base(tmp_path):
    path = tmp_path / 'tensors.safetensors'
    a = (np.arange(8, dtype=np.float32), 'F32')
    write_min(str(path), { 'x': a })
    with SafeTensorFile(str(path)) as st:
        X, _ = st.get('x')
        # NumPy view over mmapped buffer; base chain includes a memoryview on Py3.11+
        assert X.base is not None


def test_errors_bad_header_length(tmp_path):
    path = tmp_path / 'bad1.safetensors'
    # Write header_len longer than actual header to trigger error
    header = json.dumps({}).encode('utf-8')
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(header) + 10))
        f.write(header)
    with pytest.raises(ValueError):
        with SafeTensorFile(str(path)):
            pass


def test_errors_offsets_out_of_range(tmp_path):
    path = tmp_path / 'bad2.safetensors'
    # Create header where offsets exceed file size
    entries = {
        'x': { 'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 16] },
    }
    hb = json.dumps(entries).encode('utf-8')
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hb)))
        f.write(hb)
        f.write(b'\x00' * 8)  # not enough for 16 bytes
    with pytest.raises(ValueError):
        with SafeTensorFile(str(path)):
            pass


def test_errors_size_mismatch(tmp_path):
    path = tmp_path / 'bad3.safetensors'
    # shape [2,2] with F32 should be 16 bytes; lie in offsets
    entries = {
        'x': { 'dtype': 'F32', 'shape': [2, 2], 'data_offsets': [0, 8] },
    }
    hb = json.dumps(entries).encode('utf-8')
    data = b'\x00' * 16
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hb)))
        f.write(hb)
        f.write(data)
    with pytest.raises(ValueError):
        with SafeTensorFile(str(path)):
            pass


def test_errors_unsupported_dtype(tmp_path):
    path = tmp_path / 'bad4.safetensors'
    entries = {
        'x': { 'dtype': 'F128', 'shape': [1], 'data_offsets': [0, 16] },
    }
    hb = json.dumps(entries).encode('utf-8')
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hb)))
        f.write(hb)
        f.write(b'\x00' * 16)
    with pytest.raises(ValueError):
        with SafeTensorFile(str(path)):
            pass


