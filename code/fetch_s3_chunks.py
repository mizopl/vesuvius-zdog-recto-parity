#!/usr/bin/env python3
"""Stream required 128x128x128 OME-Zarr chunks from the public Vesuvius Challenge S3 bucket."""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter

CHUNK_BYTES = 128 * 128 * 128


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <base_url> <needed_chunks.txt> <out_dir> [workers=32]")
        sys.exit(1)
    base_url = sys.argv[1].rstrip("/")
    list_path = sys.argv[2]
    out_dir = sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    os.makedirs(out_dir, exist_ok=True)

    chunks = []
    with open(list_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                chunks.append((int(parts[0]), int(parts[1]), int(parts[2])))

    todo = []
    already = 0
    for iz, iy, ix in chunks:
        bin_path = os.path.join(out_dir, f"{iz}_{iy}_{ix}.bin")
        miss_path = os.path.join(out_dir, f"{iz}_{iy}_{ix}.404")
        if (os.path.exists(bin_path) and os.path.getsize(bin_path) == CHUNK_BYTES) or os.path.exists(miss_path):
            already += 1
        else:
            todo.append((iz, iy, ix))

    print(f"Total chunks in list: {len(chunks)}, already cached: {already}, to fetch: {len(todo)}", flush=True)
    if not todo:
        return

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers, max_retries=3)
    session.mount("https://", adapter)

    def fetch_one(key):
        iz, iy, ix = key
        url = f"{base_url}/{iz}/{iy}/{ix}"
        bin_path = os.path.join(out_dir, f"{iz}_{iy}_{ix}.bin")
        miss_path = os.path.join(out_dir, f"{iz}_{iy}_{ix}.404")
        for attempt in range(4):
            try:
                r = session.get(url, timeout=30)
                if r.status_code == 200:
                    data = r.content
                    if len(data) != CHUNK_BYTES:
                        raise RuntimeError(f"Expected {CHUNK_BYTES} bytes, got {len(data)}")
                    tmp = bin_path + ".tmp"
                    with open(tmp, "wb") as fh:
                        fh.write(data)
                    os.replace(tmp, bin_path)
                    return "ok"
                elif r.status_code == 404:
                    with open(miss_path, "wb") as fh:
                        pass
                    return "404"
                else:
                    time.sleep(0.5 * (attempt + 1))
            except Exception as e:
                if attempt == 3:
                    raise e
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Failed to fetch {url}")

    t0 = time.time()
    n_ok = 0
    n_404 = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, k): k for k in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res == "ok":
                n_ok += 1
            else:
                n_404 += 1
            if i % 50 == 0 or i == len(todo):
                dt = time.time() - t0
                mb = (n_ok * CHUNK_BYTES) / (1024 * 1024)
                print(
                    f"  [{i}/{len(todo)}] ok={n_ok} 404={n_404} ({mb:.1f} MiB in {dt:.1f}s, {mb/max(dt,1e-3):.1f} MiB/s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
