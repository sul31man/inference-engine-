from __future__ import annotations

import argparse
import time

import numpy as np

from safetensors.reader import SafeTensorFile


def main():
    ap = argparse.ArgumentParser(description="Inspect and micro-benchmark a .safetensors file")
    ap.add_argument('path', help='Path to .safetensors file')
    args = ap.parse_args()

    total_bytes = 0
    names = []
    with SafeTensorFile(args.path) as st:
        print(f"Opened {args.path}")
        print("First 10 keys:")
        for i, k in enumerate(st.keys()):
            if i < 10:
                info = st.meta(k)
                print(f"  {k}: dtype={info['dtype']} shape={info['shape']} bytes={info['data_offsets'][1]-info['data_offsets'][0]}")
            names.append(k)
        # micro-bench
        t0 = time.perf_counter()
        for k in names:
            arr, info = st.get(k)
            total_bytes += (info['data_offsets'][1] - info['data_offsets'][0])
            # touch a few elements so optimizer can't skip
            _ = arr.ravel()[0:1].sum()
        dt = time.perf_counter() - t0

    mb = total_bytes / (1024 * 1024)
    mbps = mb / dt if dt > 0 else float('inf')
    print(f"Read {mb:.2f} MB via zero-copy views in {dt*1000:.1f} ms -> {mbps:.1f} MB/s")


if __name__ == '__main__':
    main()


