#!/usr/bin/env python3
"""Fetch shared Zarr chunks and scan the top large auto_grown segments of PHerc1203 (>= 6.91 cm2)."""
import os
import subprocess
from pathlib import Path
import tifffile

ROOT = Path("/workspace/v3r1t4s/private/vesuvius-unread-hunt")
PY = "/opt/homebrew/bin/python3"
RENDER_EXE = str(ROOT / "code/render_tifxyz_any")
FETCH_SCRIPT = str(ROOT / "code/fetch_s3_chunks.py")
SCAN_SCRIPT = str(ROOT / "code/scan_segment.py")

S3_URL_1203 = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr/0"
)
CHUNKS_DIR = str(ROOT / "chunks/PHerc1203")

TOP_SEGS = [
    "auto_grown_20250930000321811",  # 6.91 cm2, 0.01% jumps
    "auto_grown_20250929222256117",  # 7.47 cm2, 0.28% jumps
    "auto_grown_20251005230118636",  # 7.61 cm2, 0.68% jumps
    "auto_grown_20251005221856743",  # 11.79 cm2, 1.03% jumps
]


def main():
    for seg_name in TOP_SEGS:
        s = ROOT / f"meshes/PHerc1203/{seg_name}"
        chunks_txt = str(s / "chunks.txt")
        print(f"\n=== [PHerc1203] Fetching chunks for {seg_name} ===", flush=True)
        subprocess.run([PY, FETCH_SCRIPT, S3_URL_1203, chunks_txt, CHUNKS_DIR, "32"], check=True)

        x = tifffile.imread(str(s / "x.tif"))
        rows, cols = x.shape
        zarr_out = str(ROOT / f"work/PHerc1203/{seg_name}_66.zarr")
        scan_out = str(ROOT / f"results/PHerc1203/{seg_name}")
        print(f"=== [PHerc1203] Rendering {seg_name} ({rows}x{cols} -> {rows*20}x{cols*20}) ===", flush=True)
        subprocess.run([
            RENDER_EXE, "--render",
            str(s / "x.bin"), str(s / "y.bin"), str(s / "z.bin"),
            str(rows), str(cols), "18977", "6844", "6844",
            CHUNKS_DIR, zarr_out
        ], check=True)
        print(f"=== [PHerc1203] Scanning {seg_name} on MPS ===", flush=True)
        subprocess.run([
            PY, SCAN_SCRIPT, zarr_out, str(rows), str(cols), scan_out
        ], check=True)


if __name__ == "__main__":
    main()
