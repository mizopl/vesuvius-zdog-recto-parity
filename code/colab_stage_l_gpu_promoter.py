#!/usr/bin/env python3
"""
Stage L — Google Colab GPU Ultra-Dense 1-Voxel 4x-TTA + 4-Model (`dense_native + seed42 + seed43 + Hecate`) Promoter
Runs on Google Colab (`A100` / `L4`) via `/opt/homebrew/bin/colab` to upgrade high-priority
`PHerc1203` and `PHerc0800` segments (`ag_target_patch_m7`, `ag_target_patch_m7` [neighbor of `164713356`],
`ag_target_patch_m7`, `ag_target_patch_m7`, `ag_PANORAMA_TARGET_left_m7`, `ag_PANORAMA_TARGET_right_m7`,
`ag_target_patch_m7`, `ag_target_patch_ext_m7`) to full 4-model `4x TTA` with 1-voxel sub-depths
`{d*-1, d*, d*+1}`!
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "code"))
from colab_stage_i_wide_scout import COLAB_BIN, REMOTE_SETUP_SCRIPT, run_colab
from stage_i_prepare_36_candidates import SCROLL_CONFIGS

REMOTE_STAGE_L_PROMOTER = r'''
import argparse, gc, json, os, shutil, subprocess, sys, time
from pathlib import Path
import cv2
import numpy as np
import tifffile
import torch
from scipy.ndimage import gaussian_filter

if "/content/villa/vesuvius/src" not in sys.path:
    sys.path.insert(0, "/content/villa/vesuvius/src")
if "/content/ckpt" not in sys.path:
    sys.path.insert(0, "/content/ckpt")

from vesuvius.ink_detection.inference.infer import (
    _sliding_positions_1d,
    compute_importance_map_2d,
    configure_model,
    logits_to_probabilities,
)
import hecate

CKPT_DIR = Path("/content/ckpt")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
IS_A100 = ("A100" in GPU_NAME) or ("H100" in GPU_NAME)

if "villa_models" not in globals():
    t_load = time.time()
    villa_models = {}
    for mname, fn in [
        ("dnative", "dense_native-016000.pth"),
        ("seed42", "hybrid_3d2d-seed42_step-075000.pth"),
        ("seed43", "hybrid_3d2d-seed43_step-075000.pth"),
    ]:
        cfg = configure_model(argparse.Namespace(checkpoint=CKPT_DIR / fn, amp_dtype="default"))
        p = int(cfg.patch_size)
        w_np = compute_importance_map_2d(patch_size=(p, p), mode="hann").numpy()
        villa_models[mname] = (
            cfg.model.to("cuda").eval(),
            cfg.preprocessing,
            torch.from_numpy(w_np).to("cuda", dtype=torch.float32),
            p,
        )

    hecate_model = hecate.load_model(str(CKPT_DIR / "hecate_9.6um.pth"), "cuda")
    hecate_d, hecate_p, _ = hecate_model.patch_size
    hecate_win_1d = torch.hann_window(hecate_p, periodic=False, device="cuda")
    hecate_win_2d = (hecate_win_1d[:, None] * hecate_win_1d[None, :])
    hecate_win_2d = (hecate_win_2d / hecate_win_2d.max().clamp_min(1e-7)).clamp_min(0.001)
    QUANT_TENSOR = torch.tensor([0.01, 0.50, 0.99], device="cuda", dtype=torch.float32)
    print(f"[Stage L Colab GPU] Loaded 4 models on {GPU_NAME} in {time.time()-t_load:.1f}s", flush=True)

# Import helper functions from remote_seg_runner if present
exec(open("/content/remote_base_funcs.py").read(), globals())
'''


def build_remote_script() -> str:
    from colab_stage_i_wide_scout import REMOTE_SCOUT_RUNNER
    # Lower the promotion gate in REMOTE_SCOUT_RUNNER so ALL queued Stage L candidates
    # are forcibly promoted to 4x TTA + seed42 + seed43 + 1-voxel sub-depths {d*-1, d*, d*+1}!
    script = REMOTE_SCOUT_RUNNER.replace(
        "promoted = bool(\n    max(r_best_raw, r_best_hp, r_best_thin) >= 0.24\n    and (atten_pm4 >= 30.0 or hc_atten_pm4 >= 45.0)\n)",
        "promoted = True  # Stage L Force-Promotion to 4x TTA + seed42/43 + 1-voxel {d*-1, d*, d*+1}",
    )
    return script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu", type=str, default="A100")
    args = parser.parse_args()

    target_cids = [
        ("PHerc1203", "ag_target_patch_m7"),
        ("PHerc1203", "ag_target_patch_m7"),
        ("PHerc1203", "ag_target_patch_m7"),
        ("PHerc1203", "ag_target_patch_m7"),
        ("PHerc1203", "ag_PANORAMA_TARGET_left_m7"),
        ("PHerc1203", "ag_PANORAMA_TARGET_right_m7"),
        ("PHerc1203", "ag_target_patch_m7"),
        ("PHerc0800", "ag_target_patch_ext_m7"),
    ]
    manifest = json.loads((BASE_DIR / "meshes" / "stage_i_scout_manifest.json").read_text())
    cand_map = {(m["scroll_key"], m["uuid"]): m for m in manifest["candidates"]}
    cands_list = [cand_map[k] for k in target_cids if k in cand_map]
    my_cands = [c for i, c in enumerate(cands_list) if (i % args.num_workers) == args.worker_id]

    w = args.worker_id
    session_name = f"vesuvius-l{w}"
    cfg_dir = Path(f"/tmp/colab_l{w}")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "sessions.json"
    if cfg_path.exists():
        cfg_path.unlink()

    out_root = BASE_DIR / "results_stage_i" / f"w{w}"
    out_root.mkdir(parents=True, exist_ok=True)

    tar_path = cfg_dir / f"worker_input_l{w}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(BASE_DIR / "code" / "render_tifxyz_82.cpp", arcname="code/render_tifxyz_82.cpp")
        tar.add(BASE_DIR / "code" / "fetch_s3_chunks.py", arcname="code/fetch_s3_chunks.py")
        for c in my_cands:
            sc_key = c["scroll_key"]
            cid = c["uuid"]
            seg_dir = BASE_DIR / "meshes" / "stage_i_m7snap" / sc_key / cid
            if seg_dir.exists():
                tar.add(seg_dir, arcname=f"meshes/stage_i_m7snap/{sc_key}/{cid}")

    setup_py = cfg_dir / "remote_setup.py"
    setup_py.write_text(REMOTE_SETUP_SCRIPT)
    seg_runner_py = cfg_dir / "remote_seg_runner.py"
    seg_runner_py.write_text(build_remote_script())

    res = run_colab(cfg_path, ["new", "-s", session_name, "--gpu", args.gpu], check=False)
    if res.returncode != 0 and args.gpu != "L4":
        print(f"A100 unavailable for {session_name}, falling back to L4...", flush=True)
        run_colab(cfg_path, ["new", "-s", session_name, "--gpu", "L4"], check=True)
    elif res.returncode != 0:
        raise RuntimeError(f"Failed to allocate Colab GPU for {session_name}")

    try:
        run_colab(cfg_path, ["upload", "-s", session_name, str(tar_path), "/content/worker_input.tar.gz"])
        run_colab(cfg_path, ["exec", "--timeout", "1800", "-s", session_name, "-f", str(setup_py)])

        for idx_c, c in enumerate(my_cands):
            sc_key = c["scroll_key"]
            cid = c["uuid"]
            cfg_sc = SCROLL_CONFIGS[sc_key]
            job_payload = {
                "scroll_key": sc_key,
                "cand_id": cid,
                "pool": c["pool"],
                "ct_zarr_url": cfg_sc["ct_zarr_url"],
                "shape_zyx": list(cfg_sc["shape_zyx"]),
                "um": cfg_sc["um"],
                "inward_normal_sign": c["inward_normal_sign"],
            }
            job_file = cfg_dir / "current_job.json"
            job_file.write_text(json.dumps(job_payload, indent=2))
            run_colab(cfg_path, ["upload", "-s", session_name, str(job_file), "/content/current_job.json"])

            print(f"\n[Colab Worker L{w}] ({idx_c+1}/{len(my_cands)}) Promoting {sc_key}/{cid} to 4x-TTA + seed42/43 + 1-vox ...", flush=True)
            for attempt in range(2):
                res = run_colab(cfg_path, ["exec", "--timeout", "1800", "-s", session_name, "-f", str(seg_runner_py)], check=False)
                if res.returncode == 0:
                    break
                time.sleep(3)

            local_seg_tar = cfg_dir / f"seg_{sc_key}_{cid}.tar.gz"
            if local_seg_tar.exists():
                local_seg_tar.unlink()
            dl_res = run_colab(cfg_path, ["download", "-s", session_name, "/content/seg_out.tar.gz", str(local_seg_tar)], check=False)
            if dl_res.returncode == 0 and local_seg_tar.exists():
                # Unpack directly into existing w0/w1 folder where this segment resides
                dst_root = out_root
                for w_chk in ("w0", "w1"):
                    if (BASE_DIR / "results_stage_i" / w_chk / sc_key / cid).exists():
                        dst_root = BASE_DIR / "results_stage_i" / w_chk
                        break
                with tarfile.open(local_seg_tar, "r:gz") as tf:
                    tf.extractall(path=dst_root)
                print(f"[saved] Upgraded Stage L 4x-TTA {sc_key}/{cid} -> {dst_root / sc_key / cid}", flush=True)
    finally:
        run_colab(cfg_path, ["stop", "-s", session_name], check=False)


if __name__ == "__main__":
    main()
