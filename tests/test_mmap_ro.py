import os
import io
import pytest

from utils.mmap_ro import ro_mmap


def test_ro_mmap_basic(tmp_path):
    p = tmp_path / 'x.bin'
    with open(p, 'wb') as f:
        f.write(b'abcdef')
    with ro_mmap(str(p)) as (mm, size):
        assert size == 6
        assert bytes(mm[:]) == b'abcdef'


def test_ro_mmap_missing():
    with pytest.raises(FileNotFoundError):
        with ro_mmap('does_not_exist.bin'):
            pass


