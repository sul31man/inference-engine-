"""Read-only memory mapping helper.

Why it exists / When to use:
- Use this to open large binary files without loading them into RAM.
- The OS provides pages on demand (lazy), and slicing yields zero-copy views.

Example
-------
>>> from utils.mmap_ro import ro_mmap
>>> import os
>>> path = 'tmp.bin'
>>> open(path, 'wb').write(b'hello')
5
>>> with ro_mmap(path) as (mm, size):
...     assert size == 5
...     assert mm[:5] == b'hello'
>>> os.remove(path)
"""

from __future__ import annotations

import contextlib
import mmap
import os
from typing import Iterator, Tuple


@contextlib.contextmanager
def ro_mmap(path: str) -> Iterator[Tuple[mmap.mmap, int]]:
    """Context manager that memory-maps the entire file as read-only.

    Parameters
    ----------
    path : str
        Path to the file to map.

    Yields
    ------
    (mmap_obj, size) : tuple
        A read-only ``mmap.mmap`` object and the file size in bytes.

    Notes
    -----
    - Mapping is read-only (mmap.ACCESS_READ). Modifying the buffer will fail.
    - The mapping must outlive any views created from it (e.g., numpy arrays).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    f = None
    mm = None
    try:
        f = open(path, 'rb')
        size = os.fstat(f.fileno()).st_size
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        yield mm, size
    except (OSError, ValueError) as e:
        raise IOError(f"Failed to memory-map '{path}': {e}") from e
    finally:
        # Close mapping before file so no dangling views keep the FD busy
        try:
            if mm is not None:
                mm.close()
        finally:
            if f is not None:
                f.close()


