#!/usr/bin/env python3
"""Fetch the 589 shared Zarr chunks of PHerc0800, render 66-layer surface volumes
for all 6 auto_grown segments, and run multi-orientation / multi-depth 3-model MPS ink scans.
"""
import glob
import os
import subprocess
import sys
from pathlib import Path
import tifffile

ROOT = Path("/workspace/v3r1t4s/private/vesuvius-unread-hunt")
PY = "/opt/homebrew/bin/python3"
RENDER_EXE = str(ROOT / "code/render_tifxyz_any")
FETCH_SCRIPT = str(ROOT / "code/fetch_s3_chunks.py")
SCAN_SCRIPT = str(ROOT / "code/scan_segment.py")

S3_URL_0800 = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr/0"
)
CHUNKS_TXT = str(ROOT / "meshes/PHerc0800/all_chunks.txt")
CHUNKS_DIR = str(ROOT / "chunks/PHerc0800")


def main():
    print("=== STEP 1: Fetching 589 shared Zarr chunks for PHerc0800 ===", flush=True)
    subprocess.run([PY, FETCH_SCRIPT, S3_URL_0800, CHUNKS_TXT, CHUNKS_DIR, "32"], check=True)

    segs = sorted(glob.glob(str(ROOT / "meshes/PHerc0800/202510*-on-*")))
    segs = [s for s in segs if "__mesh__" not in s]
    for s in segs:
        seg_name = os.path.basename(s).replace(".tifxyz", "")
        x = tifffile.imread(f"{s}/x.tif")
        rows, cols = x.shape
        zarr_out = str(ROOT / f"work/PHerc0800/{seg_name}_66.zarr")
        scan_out = str(ROOT / f"results/PHerc0800/{seg_name}")
        print(f"\n=== STEP 2: Rendering {seg_name} ({rows}x{cols} -> {rows*20}x{cols*20}) ===", flush=True)
        subprocess.run([
            RENDER_EXE, "--render",
            f"{s}/x.bin", f"{s}/y.bin", f"{s}/z.bin",
            str(rows), str(cols), "24298", "9867", "9867",
            CHUNKS_DIR, zarr_out
        ], check=True)
        print(f"=== STEP 3: Scanning {seg_name} on MPS ===", flush=True)
        subprocess.run([
            PY, SCAN_SCRIPT, zarr_out, str(rows), str(cols), scan_out
        ], check=True)


if __name__ == "__main__":
    main()
