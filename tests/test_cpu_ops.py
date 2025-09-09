import numpy as np
import pytest

from cpu_ops import rmsnorm, linear, silu, gelu, softmax, rope_apply


def test_ops_placeholders():
    # This file is a placeholder for your implementations. Remove/replace tests as you implement.
    with pytest.raises(NotImplementedError):
        rmsnorm(np.zeros((2, 3), dtype=np.float32), np.ones(3, dtype=np.float32))
    with pytest.raises(NotImplementedError):
        linear(np.zeros((2, 3), dtype=np.float32), np.zeros((4, 3), dtype=np.float32))
    with pytest.raises(NotImplementedError):
        silu(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(NotImplementedError):
        gelu(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(NotImplementedError):
        softmax(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(NotImplementedError):
        rope_apply(np.zeros((2, 4), dtype=np.float32), np.zeros((2, 4), dtype=np.float32), np.zeros((2,), dtype=np.float32))


