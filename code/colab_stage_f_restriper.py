#!/usr/bin/env python3
"""
Stage F (Part 2) — 3D Single-Sheet Ridge-Locking Surface Restriper on Google Colab GPU.

Solves the cylindrical `grid_wrap` "onion-ring" wrap-crossing problem on unread scrolls
(PHerc0211, PHerc0846A, PHerc0358, PHerc1545, PHerc0813, PHerc0826, PHerc0175A):
1. Renders a 112-layer normal volume (`render_tifxyz_112.cpp`, z in [-55.5 .. +55.5] voxels).
2. Computes a smooth 2D Single-Sheet Displacement Field `delta_z(y, x)` in [-18, +18] voxels
   that locks onto a single continuous bright papyrus sheet ridge and refuses to cross
   dark inter-wrap air gaps (`ct < 46`), enforcing |grad(delta_z)| <= 0.04 vox/px.
3. Resamples a sheet-locked 82-layer volume `vol82_restriped` on GPU around `z0 + delta_z(y, x)`
   so `ct_mid.png` and all 4 ink models (Villa dnative, seed42, seed43 + Hecate) stay on
   ONE flat papyrus sheet across the entire 2x2 cm prize window.
4. Evaluates fine within-sheet offsets `d in {-4, 0, +4}` voxels and downloads each completed
   segment immediately into `results_stage_f_restriped/w{k}/`.
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
COLAB_BIN = "/opt/homebrew/bin/colab"

WORKER_TASKS = {
    0: [
        {
            "scroll": "PHerc0211",
            "mesh_group": "PHerc0211",
            "orient_mode": "forward",
            "depths": [-4, 0, 4],
            "segments": [
                "grid_wrap00_z09708",
                "grid_wrap01_z09708",
            ],
        },
        {
            "scroll": "PHerc0846A",
            "mesh_group": "PHerc0846A",
            "orient_mode": "reverse",
            "depths": [-4, 0, 4],
            "segments": [
                "grid_wrap01_z05327",
                "grid_wrap00_z05327",
            ],
        },
    ],
    1: [
        {
            "scroll": "PHerc0358",
            "mesh_group": "PHerc0358",
            "orient_mode": "reverse",
            "depths": [-4, 0, 4],
            "segments": [
                "grid_wrap02_z09141",
                "grid_wrap00_z09141",
            ],
        },
        {
            "scroll": "PHerc1545",
            "mesh_group": "PHerc1545",
            "orient_mode": "forward",
            "depths": [-4, 0, 4],
            "segments": [
                "grid_wrap01_z12995",
                "grid_wrap00_z12995",
            ],
        },
    ],
    2: [
        {
            "scroll": "PHerc0813",
            "mesh_group": "PHerc0813",
            "orient_mode": "reverse",
            "depths": [-4, 0, 4],
            "segments": [
                "grid_wrap02_z06457",
                "grid_wrap00_z06457",
            ],
        },
        {
            "scroll": "PHerc0826",
            "mesh_group": "PHerc0826",
            "orient_mode": "auto",
            "depths": [-4, 0, 4],
            "segments": [
                "grid_wrap01_z08460",
                "grid_wrap02_z08460",
            ],
        },
        {
            "scroll": "PHerc0175A",
            "mesh_group": "PHerc0175A",
            "orient_mode": "auto",
            "depths": [-4, 0, 4],
            "segments": [
                "grid_wrap01_z07903",
            ],
        },
    ],
}

REMOTE_SETUP_SCRIPT = r'''
import json, os, subprocess, sys, tarfile
from pathlib import Path

print("=== STAGE F 3D SINGLE-SHEET RESTRIPER COLAB GPU SETUP ===", flush=True)
subprocess.run(["nvidia-smi"], check=False)

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "zarr>=3.0.0", "pynrrd", "imagecodecs", "tifffile", "safetensors", "huggingface_hub", "einops", "opencv-python-headless", "scipy"
], check=True)

if not os.path.exists("/content/input_unpacked"):
    with tarfile.open("/content/worker_input.tar.gz", "r:gz") as tar:
        tar.extractall("/content")
    Path("/content/input_unpacked").touch()

subprocess.run([
    "g++", "-O3", "-std=c++17", "-pthread",
    "/content/code/render_tifxyz_112.cpp", "-o", "/content/code/render_tifxyz_112"
], check=True)

if not os.path.exists("/content/villa"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ScrollPrize/villa.git", "/content/villa"], check=True)

os.makedirs("/content/ckpt", exist_ok=True)
os.makedirs("/content/results", exist_ok=True)

import torch
from safetensors.torch import load_file

CKPTS = [
    ("dense_native_016000.safetensors", "https://huggingface.co/Nieuwlaar/ink9um-dense-native/resolve/main/weights/dense_native_016000.safetensors"),
    ("train_dense_native.json", "https://huggingface.co/Nieuwlaar/ink9um-dense-native/resolve/main/training/configs/train_dense_native.json"),
    ("hybrid_3d2d-seed42_step-075000.pth", "https://huggingface.co/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed42/step-075000.pth"),
    ("hybrid_3d2d-seed43_step-075000.pth", "https://huggingface.co/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed43/step-075000.pth"),
    ("hecate_9.6um.pth", "https://huggingface.co/scrollprize/hecate/resolve/main/hecate_9.6um.pth"),
    ("hecate.py", "https://huggingface.co/scrollprize/hecate/resolve/main/hecate.py"),
]
for fn, url in CKPTS:
    dst = f"/content/ckpt/{fn}"
    if not os.path.exists(dst):
        print(f"Downloading {fn} ...", flush=True)
        subprocess.run(["curl", "--http1.1", "-fSL", "--retry", "8", "--retry-all-errors", "-o", dst, url], check=True)

if not os.path.exists("/content/ckpt/dense_native-016000.pth"):
    torch.save({
        "model": load_file("/content/ckpt/dense_native_016000.safetensors"),
        "config": json.load(open("/content/ckpt/train_dense_native.json")),
        "step": 16000
    }, "/content/ckpt/dense_native-016000.pth")

open("/content/villa/vesuvius/src/vesuvius/__init__.py", "w").write("")
print("REMOTE_SETUP_COMPLETE", flush=True)
'''

REMOTE_SEGMENT_RUNNER = r'''
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
    print(f"Loaded 4 models on GPU in {time.time()-t_load:.1f}s", flush=True)
else:
    print("Reusing 4 GPU models already in kernel memory", flush=True)


def restripe_single_sheet_volume_cuda(vol112: np.ndarray, max_shift: int = 18):
    """
    Given a 112-layer normal volume vol112[Z=112, H, W] centered at z0=55.5:
    1. Computes a smooth 2D Single-Sheet Displacement Field `delta_z(y, x)` in [-max_shift, +max_shift]
       that tracks a single bright papyrus sheet ridge and refuses to cross dark air gaps (ct < 46).
    2. Resamples an 82-layer volume `vol82_restriped[82, H, W]` centered on `z0 + delta_z(y, x)`
       using sub-voxel linear interpolation along z on the GPU.
    Returns (vol82_restriped, delta_z_full, stats_dict).
    """
    t0 = time.time()
    Z, H, W = vol112.shape
    z_center = 55  # nominal center slice in 0..111

    ds = 8
    sub_ct = vol112[:, ::ds, ::ds].astype(np.float32)  # (112, Hc, Wc)
    _, Hc, Wc = sub_ct.shape

    deltas = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
    n_d = len(deltas)
    pot = np.zeros((n_d, Hc, Wc), dtype=np.float32)
    gap_pen = np.zeros((n_d, Hc, Wc), dtype=np.float32)

    for idx_d, dz in enumerate(deltas):
        zc = z_center + int(dz)
        slab5 = sub_ct[zc - 2:zc + 3]
        mean_ct = slab5.mean(axis=0)
        slab7 = sub_ct[zc - 3:zc + 4]
        is_gap = ((slab7 > 0) & (slab7 < 46)).astype(np.float32).mean(axis=0)
        gap_pen[idx_d] = gaussian_filter(is_gap, sigma=2.5)

        dz_grad = np.abs(sub_ct[zc + 2] - sub_ct[zc - 2]) * 0.25
        density_score = np.clip((mean_ct - 48.0) / 85.0, -0.5, 1.4)
        ridge_center_bonus = -np.clip(dz_grad / 45.0, 0.0, 0.8)
        prior_bonus = -0.18 * ((dz / float(max_shift)) ** 2)
        pot[idx_d] = gaussian_filter(density_score + ridge_center_bonus - 3.2 * gap_pen[idx_d], sigma=2.0) + prior_bonus

    dz_map = np.zeros((Hc, Wc), dtype=np.float32)
    delta_grid = deltas[:, None, None].astype(np.float32)

    for _ in range(18):
        dz_smooth = gaussian_filter(dz_map, sigma=3.5)
        dist_pen = -0.45 * ((delta_grid - dz_smooth[None, :, :]) ** 2)
        loc_logits = (pot + dist_pen) * 4.5
        loc_logits -= loc_logits.max(axis=0, keepdims=True)
        w = np.exp(np.clip(loc_logits, -25.0, 0.0))
        w /= np.maximum(w.sum(axis=0, keepdims=True), 1e-6)
        dz_next = np.sum(w * delta_grid, axis=0)
        dz_map = gaussian_filter(0.35 * dz_map + 0.65 * dz_next, sigma=5.0)
        dz_map = np.clip(dz_map, -float(max_shift), float(max_shift))

    dz_coarse = gaussian_filter(dz_map, sigma=9.0).astype(np.float32)
    dz_full = cv2.resize(dz_coarse, (W, H), interpolation=cv2.INTER_CUBIC)
    dz_full = np.clip(dz_full, -float(max_shift), float(max_shift))

    vol82_out = np.zeros((82, H, W), dtype=np.uint8)
    strip_h = 512
    z_idx_base = torch.arange(82, device="cuda", dtype=torch.float32)[:, None, None] + 15.0

    with torch.no_grad():
        for y0 in range(0, H, strip_h):
            y1 = min(H, y0 + strip_h)
            v_strip = torch.from_numpy(vol112[:, y0:y1, :]).to("cuda", dtype=torch.float32)
            dz_strip = torch.from_numpy(dz_full[y0:y1, :]).to("cuda", dtype=torch.float32)[None, :, :]
            z_sample = (z_idx_base + dz_strip).clamp(0.0, float(Z - 1.001))
            z0_i = z_sample.long()
            z1_i = (z0_i + 1).clamp(max=Z - 1)
            fz = z_sample - z0_i.float()
            val0 = torch.gather(v_strip, 0, z0_i)
            val1 = torch.gather(v_strip, 0, z1_i)
            interp = (val0 + fz * (val1 - val0)).round().clamp(0.0, 255.0).to(torch.uint8)
            vol82_out[:, y0:y1, :] = interp.cpu().numpy()
            del v_strip, dz_strip, z_sample, z0_i, z1_i, fz, val0, val1, interp
    torch.cuda.empty_cache()

    restripe_stats = {
        "dz_min": round(float(dz_full.min()), 2),
        "dz_max": round(float(dz_full.max()), 2),
        "dz_std": round(float(dz_full.std()), 2),
        "elapsed_s": round(time.time() - t0, 2),
    }
    print(f"  [Restripe] Locked single-sheet surface in {restripe_stats['elapsed_s']}s (dz range: {restripe_stats['dz_min']:+.1f}..{restripe_stats['dz_max']:+.1f} vox, std={restripe_stats['dz_std']:.2f})", flush=True)
    return vol82_out, dz_full, restripe_stats


def eval_villa_all3_cuda(vol17: np.ndarray, supp_2d: np.ndarray):
    """Run dnative (64x64, stride 32) and fused seed42+seed43 (128x128, stride 64) on GPU."""
    Vh, Vw = vol17.shape[1], vol17.shape[2]
    vol17_gpu = torch.from_numpy(vol17).to("cuda", non_blocking=True)
    supp_gpu = torch.from_numpy(supp_2d).to("cuda", non_blocking=True)

    t0 = time.time()
    m_dn_model, _, w_dn, p_dn = villa_models["dnative"]
    ys_dn = _sliding_positions_1d(Vh, p_dn, p_dn // 2)
    xs_dn = _sliding_positions_1d(Vw, p_dn, p_dn // 2)
    coords_dn = []
    for y0 in ys_dn:
        vh = min(p_dn, Vh - y0)
        for x0 in xs_dn:
            vw = min(p_dn, Vw - x0)
            if supp_2d[y0:y0 + vh, x0:x0 + vw].any():
                coords_dn.append((y0, x0, vh, vw))

    prob_dn = torch.zeros((Vh, Vw), dtype=torch.float32, device="cuda")
    wsum_dn = torch.zeros((Vh, Vw), dtype=torch.float32, device="cuda")
    bs_dn = 64
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(coords_dn), bs_dn):
            bc = coords_dn[start:start + bs_dn]
            B = len(bc)
            patches = torch.zeros((B, 17, p_dn, p_dn), dtype=torch.float32, device="cuda")
            supps = torch.zeros((B, p_dn, p_dn), dtype=torch.float32, device="cuda")
            for bi, (y0, x0, vh, vw) in enumerate(bc):
                sub = vol17_gpu[:, y0:y0 + vh, x0:x0 + vw].to(torch.float32)
                s = supp_gpu[y0:y0 + vh, x0:x0 + vw]
                patches[bi, :, :vh, :vw] = sub
                supps[bi, :vh, :vw] = s.to(torch.float32)
                if not s.all():
                    fill_val = sub[:, s].mean().round()
                    patches[bi] = torch.where(supps[bi].unsqueeze(0) > 0.5, patches[bi], fill_val)
            x_batch = (patches * (1.0 / 255.0)).unsqueeze(1)
            probs = logits_to_probabilities(m_dn_model(x_batch), image_hw=(p_dn, p_dn)).float()[:, 0]
            w_batch = w_dn.unsqueeze(0) * supps
            for bi, (y0, x0, vh, vw) in enumerate(bc):
                w = w_batch[bi, :vh, :vw]
                prob_dn[y0:y0 + vh, x0:x0 + vw] += probs[bi, :vh, :vw] * w
                wsum_dn[y0:y0 + vh, x0:x0 + vw] += w

    out_dn = torch.where(wsum_dn > 1e-6, prob_dn / wsum_dn.clamp_min(1e-6), torch.zeros_like(prob_dn))
    m_dn_u8 = (out_dn.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    del prob_dn, wsum_dn, out_dn
    t_dn = time.time() - t0

    t1 = time.time()
    m_42_model, _, w_128, p_128 = villa_models["seed42"]
    m_43_model, _, _, _ = villa_models["seed43"]
    ys_128 = _sliding_positions_1d(Vh, p_128, p_128 // 2)
    xs_128 = _sliding_positions_1d(Vw, p_128, p_128 // 2)
    coords_128 = []
    for y0 in ys_128:
        vh = min(p_128, Vh - y0)
        for x0 in xs_128:
            vw = min(p_128, Vw - x0)
            if supp_2d[y0:y0 + vh, x0:x0 + vw].any():
                coords_128.append((y0, x0, vh, vw))

    prob_42 = torch.zeros((Vh, Vw), dtype=torch.float32, device="cuda")
    prob_43 = torch.zeros((Vh, Vw), dtype=torch.float32, device="cuda")
    wsum_128 = torch.zeros((Vh, Vw), dtype=torch.float32, device="cuda")
    bs_128 = 32
    with torch.inference_mode():
        for start in range(0, len(coords_128), bs_128):
            bc = coords_128[start:start + bs_128]
            B = len(bc)
            patches = torch.zeros((B, 17, p_128, p_128), dtype=torch.float32, device="cuda")
            supps = torch.zeros((B, p_128, p_128), dtype=torch.float32, device="cuda")
            for bi, (y0, x0, vh, vw) in enumerate(bc):
                sub = vol17_gpu[:, y0:y0 + vh, x0:x0 + vw].to(torch.float32)
                s = supp_gpu[y0:y0 + vh, x0:x0 + vw]
                patches[bi, :, :vh, :vw] = sub
                supps[bi, :vh, :vw] = s.to(torch.float32)
                if not s.all():
                    fill_val = sub[:, s].mean().round()
                    patches[bi] = torch.where(supps[bi].unsqueeze(0) > 0.5, patches[bi], fill_val)

            sub_gpu = patches[:, :, ::2, ::2].reshape(B, -1)
            q = torch.quantile(sub_gpu, QUANT_TENSOR, dim=1)
            lo = q[0].view(B, 1, 1, 1)
            med = q[1].view(B, 1, 1, 1)
            hi = q[2].view(B, 1, 1, 1)
            patches.clamp_(lo, hi)
            sub_c = torch.clamp(sub_gpu, q[0][:, None], q[2][:, None])
            mad = torch.median(torch.abs(sub_c - q[1][:, None]), dim=1).values.view(B, 1, 1, 1)
            scale = torch.clamp(1.4826 * mad, min=1e-6)
            patches.sub_(med).div_(scale)

            x_batch = patches.unsqueeze(1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                p42 = logits_to_probabilities(m_42_model(x_batch), image_hw=(p_128, p_128)).float()[:, 0]
                p43 = logits_to_probabilities(m_43_model(x_batch), image_hw=(p_128, p_128)).float()[:, 0]

            w_batch = w_128.unsqueeze(0) * supps
            for bi, (y0, x0, vh, vw) in enumerate(bc):
                w = w_batch[bi, :vh, :vw]
                prob_42[y0:y0 + vh, x0:x0 + vw] += p42[bi, :vh, :vw] * w
                prob_43[y0:y0 + vh, x0:x0 + vw] += p43[bi, :vh, :vw] * w
                wsum_128[y0:y0 + vh, x0:x0 + vw] += w

    out_42 = torch.where(wsum_128 > 1e-6, prob_42 / wsum_128.clamp_min(1e-6), torch.zeros_like(prob_42))
    out_43 = torch.where(wsum_128 > 1e-6, prob_43 / wsum_128.clamp_min(1e-6), torch.zeros_like(prob_43))
    m_42_u8 = (out_42.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    m_43_u8 = (out_43.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    del vol17_gpu, supp_gpu, prob_42, prob_43, wsum_128, out_42, out_43
    torch.cuda.empty_cache()
    t_128 = time.time() - t1
    return m_dn_u8, m_42_u8, m_43_u8, t_dn, t_128


def eval_hecate_seamless_cuda(vol66: np.ndarray, supp_2d: np.ndarray, reverse: bool, bs: int = 48):
    """GPU-resident Hann-blended Hecate inference at stride = 32."""
    t0 = time.time()
    depth, H, W = vol66.shape
    d, p = hecate_d, hecate_p
    stride = p // 2
    z0 = depth // 2 - d // 2
    zs = slice(depth - z0 - d, depth - z0) if reverse else slice(z0, z0 + d)
    sub48 = vol66[zs][::-1].copy() if reverse else vol66[zs].copy()

    sub48_gpu = torch.from_numpy(sub48).to("cuda", non_blocking=True)
    supp_gpu = torch.from_numpy(supp_2d).to("cuda", non_blocking=True)

    ys = _sliding_positions_1d(H, p, stride)
    xs = _sliding_positions_1d(W, p, stride)
    coords = []
    for y0 in ys:
        ph = min(p, H - y0)
        for x0 in xs:
            pw = min(p, W - x0)
            if supp_2d[y0:y0 + ph, x0:x0 + pw].any():
                coords.append((y0, x0, ph, pw))

    prob_sum = torch.zeros((H, W), dtype=torch.float32, device="cuda")
    weight_sum = torch.zeros((H, W), dtype=torch.float32, device="cuda")
    divisor = float(hecate_model.divisor)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(coords), bs):
            bc = coords[start:start + bs]
            B = len(bc)
            patches = torch.zeros((B, d, p, p), dtype=torch.float32, device="cuda")
            valid_batch = torch.zeros((B, p, p), dtype=torch.bool, device="cuda")
            for bi, (y0, x0, ph, pw) in enumerate(bc):
                raw = sub48_gpu[:, y0:y0 + ph, x0:x0 + pw].to(torch.float32)
                s = supp_gpu[y0:y0 + ph, x0:x0 + pw]
                patches[bi, :, :ph, :pw] = raw
                valid_batch[bi, :ph, :pw] = s
                if not valid_batch[bi].all():
                    fill_val = raw[:, s].mean().round()
                    patches[bi] = torch.where(valid_batch[bi].unsqueeze(0), patches[bi], fill_val)

            image_gpu = (patches * (1.0 / divisor)).unsqueeze(1)
            valid_gpu = valid_batch[:, None, None].expand(-1, 1, d, -1, -1)
            probs = hecate_model(image_gpu, valid_gpu).float().sigmoid()[:, 0]
            w_batch = hecate_win_2d.unsqueeze(0) * valid_batch.to(torch.float32)
            for bi, (y0, x0, ph, pw) in enumerate(bc):
                w = w_batch[bi, :ph, :pw]
                prob_sum[y0:y0 + ph, x0:x0 + pw] += probs[bi, :ph, :pw] * w
                weight_sum[y0:y0 + ph, x0:x0 + pw] += w

    out = torch.where(weight_sum > 1e-6, prob_sum / weight_sum.clamp_min(1e-6), torch.zeros_like(prob_sum))
    m_hc_u8 = torch.round(out.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    del sub48_gpu, supp_gpu, prob_sum, weight_sum, out
    torch.cuda.empty_cache()
    return m_hc_u8, time.time() - t0


def compute_zscore_ensemble_and_gold(ct_mid: np.ndarray, m_dn: np.ndarray, m_hc: np.ndarray,
                                     m_42: np.ndarray, m_43: np.ndarray, valid_mask: np.ndarray):
    dn = m_dn.astype(np.float32) / 255.0
    hc = m_hc.astype(np.float32) / 255.0
    s42 = m_42.astype(np.float32) / 255.0
    s43 = m_43.astype(np.float32) / 255.0

    def znorm(arr):
        vals = arr[valid_mask]
        if vals.size < 100:
            return np.zeros_like(arr)
        mu = float(vals.mean())
        sd = max(float(vals.std()), 1e-4)
        return (arr - mu) / sd

    ens_z = (znorm(dn) + znorm(hc) + znorm(s42) + znorm(s43)) / 4.0
    z_vals = ens_z[valid_mask]
    if z_vals.size > 100:
        lo_z, hi_z = np.percentile(z_vals, [1.0, 99.9])
        hi_z = max(float(hi_z), float(lo_z) + 1.0)
    else:
        lo_z, hi_z = -0.718, 3.863
    ens_u8 = (np.clip((ens_z - lo_z) / (hi_z - lo_z), 0.0, 1.0) * 255.0).astype(np.uint8)
    ens_u8[~valid_mask] = 0

    alpha = 0.9 * np.clip((ens_u8.astype(np.float32) / 255.0 - 0.35) / 0.40, 0.0, 1.0)
    alpha[~valid_mask] = 0.0
    ct_f = ct_mid.astype(np.float32)[..., None]
    gold = np.array([255.0, 235.0, 120.0], dtype=np.float32)[None, None, :]
    rgb = np.clip(ct_f * 0.45 * (1.0 - alpha[..., None]) + gold * alpha[..., None], 0.0, 255.0).astype(np.uint8)
    return ens_z, ens_u8, rgb


def max_window_correlation(villa_f: np.ndarray, hecate_f: np.ndarray, valid_mask: np.ndarray, um: float):
    win_px = int(round(20000.0 / um))
    H, W = villa_f.shape
    if H < win_px or W < win_px:
        v = valid_mask
        if v.sum() < 1000:
            return 0.0, [0, 0, H, W]
        a, b = villa_f[v], hecate_f[v]
        if a.std() < 1e-4 or b.std() < 1e-4:
            return 0.0, [0, 0, H, W]
        return float(np.corrcoef(a, b)[0, 1]), [0, 0, H, W]

    best_r = -1.0
    best_box = [0, 0, win_px, win_px]
    ys = sorted(set(list(range(0, H - win_px + 1, 200)) + [H - win_px]))
    xs = sorted(set(list(range(0, W - win_px + 1, 200)) + [W - win_px]))
    for y0 in ys:
        for x0 in xs:
            v = valid_mask[y0:y0 + win_px, x0:x0 + win_px]
            if v.mean() < 0.70:
                continue
            a = villa_f[y0:y0 + win_px, x0:x0 + win_px][v]
            b = hecate_f[y0:y0 + win_px, x0:x0 + win_px][v]
            if a.std() < 1e-4 or b.std() < 1e-4:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            if r > best_r:
                best_r = r
                best_box = [int(y0), int(x0), int(win_px), int(win_px)]
    return best_r, best_box


manifest = json.load(open("/content/manifest_stage_d.json"))
job = json.load(open("/content/current_job.json"))
scroll = job["scroll"]
mesh_group = job["mesh_group"]
seg_name = job["seg_name"]
orient_mode = job.get("orient_mode", "auto")
depths = job.get("depths", [-4, 0, 4])

info = manifest[scroll]
s3_scroll = info.get("s3_scroll", scroll)
vol_name = info["volume"]
um = float(info["um"])
volZ, volY, volX = info["shape_zyx"]
s3_vol_url = f"https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/{s3_scroll}/volumes/{vol_name}/0"
chunks_dir = f"/content/chunks/{scroll}"
os.makedirs(chunks_dir, exist_ok=True)

m_dir = Path(f"/content/meshes/{mesh_group}/{seg_name}")
t_seg = time.time()
print(f"--- Starting Stage F 3D Single-Sheet Restriper on {scroll}/{seg_name} ---", flush=True)
meta = json.load(open(m_dir / "meta.json"))
x = tifffile.imread(str(m_dir / "x.tif")).astype(np.float32)
y = tifffile.imread(str(m_dir / "y.tif")).astype(np.float32)
z = tifffile.imread(str(m_dir / "z.tif")).astype(np.float32)
rows, cols = x.shape
x.tofile(str(m_dir / "x.bin"))
y.tofile(str(m_dir / "y.bin"))
z.tofile(str(m_dir / "z.bin"))
chunks_txt = str(m_dir / "chunks_112.txt")

subprocess.run([
    "/content/code/render_tifxyz_112", "--list-chunks",
    str(m_dir / "x.bin"), str(m_dir / "y.bin"), str(m_dir / "z.bin"),
    str(rows), str(cols), str(volZ), str(volY), str(volX), chunks_txt
], check=True)
subprocess.run(["python3", "/content/code/fetch_s3_chunks.py", s3_vol_url, chunks_txt, chunks_dir, "64"], check=True)

zarr_dir = Path(f"/content/work/{scroll}/{seg_name}_112.zarr")
subprocess.run([
    "/content/code/render_tifxyz_112", "--render",
    str(m_dir / "x.bin"), str(m_dir / "y.bin"), str(m_dir / "z.bin"),
    str(rows), str(cols), str(volZ), str(volY), str(volX), chunks_dir, str(zarr_dir)
], check=True)

H, W = rows * 20, cols * 20
vol112 = np.zeros((112, H, W), dtype=np.uint8)
tilesY, tilesX = (H + 127) // 128, (W + 127) // 128
for ty in range(tilesY):
    y0, dy = ty * 128, min(128, H - ty * 128)
    for tx in range(tilesX):
        x0, dx = tx * 128, min(128, W - tx * 128)
        cp = zarr_dir / "0" / "0" / str(ty) / str(tx)
        if cp.exists():
            buf = np.fromfile(str(cp), dtype=np.uint8).reshape(112, 128, 128)
            vol112[:, y0:y0 + dy, x0:x0 + dx] = buf[:, :dy, :dx]
shutil.rmtree(zarr_dir, ignore_errors=True)
print(f"  Rendered 112-layer volume ({H}x{W}) in {time.time()-t_seg:.1f}s", flush=True)

# Run 3D Single-Sheet Ridge-Locking Surface Restriper!
vol82, dz_full, restripe_stats = restripe_single_sheet_volume_cuda(vol112, max_shift=18)
del vol112
gc.collect()

out_dir = Path(f"/content/results/{scroll}/{seg_name}")
out_dir.mkdir(parents=True, exist_ok=True)

# Save delta_z visualization map
dz_vis = np.clip((dz_full + 18.0) / 36.0 * 255.0, 0.0, 255.0).astype(np.uint8)
cv2.imwrite(str(out_dir / "delta_z_sheet_map.png"), dz_vis)

valid_mask = (vol82[40] > 0)
area_cm2 = round(float(valid_mask.sum()) * (um * 1e-4) ** 2, 2)


def run_single_depth(vol66: np.ndarray, orient: str, label: str = ""):
    rev_flag = (orient == "reverse")
    z0 = 26 if rev_flag else 23
    sub17 = vol66[z0:z0 + 17]
    sub17 = sub17[::-1].copy() if rev_flag else sub17.copy()
    supp_2d = sub17.any(axis=0)
    m_dn, m_42, m_43, t_dn, t_128 = eval_villa_all3_cuda(sub17, supp_2d)
    print(f"    [{label} {orient}] villa_dn={t_dn:.1f}s villa_128={t_128:.1f}s ...", end="", flush=True)
    m_hc, t_hc = eval_hecate_seamless_cuda(vol66, supp_2d, reverse=rev_flag)
    print(f" hecate={t_hc:.1f}s", flush=True)

    ct_slice = vol66[34].copy()
    ens_z_f, ens_z_u8, gold_rgb = compute_zscore_ensemble_and_gold(ct_slice, m_dn, m_hc, m_42, m_43, valid_mask)
    villa_f = (m_dn.astype(np.float32) + m_42.astype(np.float32) + m_43.astype(np.float32)) / 3.0
    hecate_f = m_hc.astype(np.float32)
    v_v, v_h = villa_f[valid_mask], hecate_f[valid_mask]
    r_vh = float(np.corrcoef(v_v, v_h)[0, 1]) if (v_v.size > 500 and v_v.std() > 1e-3 and v_h.std() > 1e-3) else 0.0
    win_r, win_box = max_window_correlation(villa_f, hecate_f, valid_mask, um)
    ens4 = 0.25 * (m_dn.astype(np.float32) + m_42.astype(np.float32) + m_43.astype(np.float32) + m_hc.astype(np.float32))
    p50, p995 = [float(x) for x in np.percentile(ens4[valid_mask], [50.0, 99.5])] if v_v.size > 500 else (0.0, 0.0)
    hc_p995 = float(np.percentile(v_h, 99.5)) if v_h.size > 500 else 0.0
    return {
        "m_dn": m_dn, "m_42": m_42, "m_43": m_43, "m_hc": m_hc,
        "ct_slice": ct_slice, "ens_z_f": ens_z_f, "ens_z_u8": ens_z_u8, "gold_rgb": gold_rgb,
        "villa_f": villa_f, "hecate_f": hecate_f,
        "stats": {
            "ens4_p50": round(p50, 2),
            "ens4_p995": round(p995, 2),
            "ens4_contrast": round(p995 - p50, 2),
            "hecate_p995": round(hc_p995, 2),
            "r_villa_hecate": round(r_vh, 4),
            "max_2x2cm_r_villa_hecate": round(win_r, 4),
            "max_2x2cm_box_yxwh": win_box,
        }
    }


chosen_orient = orient_mode
d0_cache = {}
if orient_mode == "auto":
    vol_d0 = vol82[8:74]
    res_rev = run_single_depth(vol_d0, "reverse", label="auto-d0")
    res_fwd = run_single_depth(vol_d0, "forward", label="auto-d0")
    d0_cache["reverse"] = res_rev
    d0_cache["forward"] = res_fwd
    score_rev = max(res_rev["stats"]["r_villa_hecate"], res_rev["stats"]["max_2x2cm_r_villa_hecate"])
    score_fwd = max(res_fwd["stats"]["r_villa_hecate"], res_fwd["stats"]["max_2x2cm_r_villa_hecate"])
    chosen_orient = "reverse" if score_rev >= score_fwd else "forward"
    other_orient = "forward" if chosen_orient == "reverse" else "reverse"
    cv2.imwrite(str(out_dir / f"ens_z_{other_orient}_d+0.png"), d0_cache[other_orient]["ens_z_u8"])
    cv2.imwrite(str(out_dir / f"submission_gold_{other_orient}_d+0.png"), cv2.cvtColor(d0_cache[other_orient]["gold_rgb"], cv2.COLOR_RGB2BGR))
    print(f"  [{scroll}/{seg_name}] Auto-orient picked '{chosen_orient}' (rev={score_rev:.4f} vs fwd={score_fwd:.4f})", flush=True)

depth_results = {}
dn_stack, s42_stack, s43_stack, hc_stack, ct_stack, z_stack = [], [], [], [], [], []
for d in depths:
    t_d = time.time()
    if d == 0 and chosen_orient in d0_cache:
        res_d = d0_cache[chosen_orient]
    else:
        s0 = 8 + d
        vol_d = vol82[s0:s0 + 66]
        res_d = run_single_depth(vol_d, chosen_orient, label=f"d={d:+d}")
    depth_results[f"d{d:+d}"] = res_d["stats"]
    cv2.imwrite(str(out_dir / f"ens_z_{chosen_orient}_d{d:+d}.png"), res_d["ens_z_u8"])
    cv2.imwrite(str(out_dir / f"submission_gold_{chosen_orient}_d{d:+d}.png"), cv2.cvtColor(res_d["gold_rgb"], cv2.COLOR_RGB2BGR))
    if d == 0:
        cv2.imwrite(str(out_dir / "ct_mid.png"), res_d["ct_slice"])
        cv2.imwrite(str(out_dir / f"dnative_{chosen_orient}_d+0.png"), res_d["m_dn"])
        cv2.imwrite(str(out_dir / f"hecate_{chosen_orient}_d+0.png"), res_d["m_hc"])
    dn_stack.append(res_d["m_dn"].astype(np.float32))
    s42_stack.append(res_d["m_42"].astype(np.float32))
    s43_stack.append(res_d["m_43"].astype(np.float32))
    hc_stack.append(res_d["m_hc"].astype(np.float32))
    ct_stack.append(res_d["ct_slice"].astype(np.float32))
    z_stack.append(res_d["ens_z_f"])
    st_i = res_d["stats"]
    print(f"  [{scroll}/{seg_name} {chosen_orient} d={d:+d}] r_all={st_i['r_villa_hecate']:.4f} r_2x2cm={st_i['max_2x2cm_r_villa_hecate']:.4f} ens4_p99.5={st_i['ens4_p995']:.1f} hc_p99.5={st_i['hecate_p995']:.1f} ({time.time()-t_d:.1f}s)", flush=True)

z_arr = np.stack(z_stack, axis=0)
local_energy = np.stack([gaussian_filter(np.maximum(z_d, 0.0), sigma=20.0) for z_d in z_arr], axis=0)
prior = np.array([np.exp(-0.5 * (d / 6.0) ** 2) for d in depths], dtype=np.float32)[:, None, None]
logits_w = 2.0 * local_energy + np.log(prior)
logits_w -= logits_w.max(axis=0, keepdims=True)
w_depth = np.exp(logits_w)
w_depth /= np.maximum(w_depth.sum(axis=0, keepdims=True), 1e-6)

dn_md = (w_depth * np.stack(dn_stack, axis=0)).sum(axis=0).astype(np.uint8)
s42_md = (w_depth * np.stack(s42_stack, axis=0)).sum(axis=0).astype(np.uint8)
s43_md = (w_depth * np.stack(s43_stack, axis=0)).sum(axis=0).astype(np.uint8)
hc_md = (w_depth * np.stack(hc_stack, axis=0)).sum(axis=0).astype(np.uint8)
ct_md = (w_depth * np.stack(ct_stack, axis=0)).sum(axis=0).astype(np.uint8)

ens_z_md_f, ens_z_md_u8, gold_md_rgb = compute_zscore_ensemble_and_gold(ct_md, dn_md, hc_md, s42_md, s43_md, valid_mask)
villa_md_f = (dn_md.astype(np.float32) + s42_md.astype(np.float32) + s43_md.astype(np.float32)) / 3.0
hecate_md_f = hc_md.astype(np.float32)
v_v, v_h = villa_md_f[valid_mask], hecate_md_f[valid_mask]
r_md = float(np.corrcoef(v_v, v_h)[0, 1]) if (v_v.size > 500 and v_v.std() > 1e-3 and v_h.std() > 1e-3) else 0.0
win_r_md, win_box_md = max_window_correlation(villa_md_f, hecate_md_f, valid_mask, um)
ens4_md = 0.25 * (dn_md.astype(np.float32) + s42_md.astype(np.float32) + s43_md.astype(np.float32) + hc_md.astype(np.float32))
p50_md, p995_md = [float(x) for x in np.percentile(ens4_md[valid_mask], [50.0, 99.5])] if v_v.size > 500 else (0.0, 0.0)
hc_p995_md = float(np.percentile(v_h, 99.5)) if v_h.size > 500 else 0.0

cv2.imwrite(str(out_dir / f"ens_z_{chosen_orient}_multidepth.png"), ens_z_md_u8)
cv2.imwrite(str(out_dir / f"submission_gold_{chosen_orient}_multidepth.png"), cv2.cvtColor(gold_md_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(out_dir / f"dnative_{chosen_orient}_multidepth.png"), dn_md)
cv2.imwrite(str(out_dir / f"hecate_{chosen_orient}_multidepth.png"), hc_md)

by, bx, bh, bw = win_box_md
cv2.imwrite(str(out_dir / f"crop_2x2cm_gold_{chosen_orient}.png"), cv2.cvtColor(gold_md_rgb[by:by+bh, bx:bx+bw], cv2.COLOR_RGB2BGR))
cv2.imwrite(str(out_dir / f"crop_2x2cm_ens_z_{chosen_orient}.png"), ens_z_md_u8[by:by+bh, bx:bx+bw])

seg_report = {
    "scroll": scroll,
    "segment": seg_name,
    "um": um,
    "shape_hw": [H, W],
    "area_cm2": area_cm2,
    "chosen_orient": chosen_orient,
    "restripe_stats": restripe_stats,
    "depths": depth_results,
    "multidepth": {
        "ens4_p50": round(p50_md, 2),
        "ens4_p995": round(p995_md, 2),
        "ens4_contrast": round(p995_md - p50_md, 2),
        "hecate_p995": round(hc_p995_md, 2),
        "r_villa_hecate": round(r_md, 4),
        "max_2x2cm_r_villa_hecate": round(win_r_md, 4),
        "max_2x2cm_box_yxwh": win_box_md,
    },
    "elapsed_s": round(time.time() - t_seg, 1),
    "meta": meta,
}
with open(out_dir / "metrics_stage_d.json", "w") as f:
    json.dump(seg_report, f, indent=2)

print(f"  ==> [{scroll}/{seg_name} RESTRIPED MULTIDEPTH] r_all={r_md:.4f} r_2x2cm={win_r_md:.4f} ens4_p99.5={p995_md:.1f} hc_p99.5={hc_p995_md:.1f} ({seg_report['elapsed_s']}s total)", flush=True)
subprocess.run(["tar", "-czf", "/content/seg_out.tar.gz", "-C", "/content/results", f"{scroll}/{seg_name}"], check=True)
print("SEGMENT_COMPLETE_OK", flush=True)
'''


def build_manifest_stage_d():
    return json.loads((BASE_DIR / "manifest_stage_d.json").read_text())


def run_colab(config_path: Path, args: list, check: bool = True):
    cmd = [COLAB_BIN, "--config", str(config_path)] + args
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--gpu", type=str, default="A100")
    args = parser.parse_args()

    w = args.worker_id
    tasks = WORKER_TASKS[w]
    session_name = f"vesuvius-f{w}"
    cfg_dir = Path(f"/tmp/colab_f{w}")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "sessions.json"
    if cfg_path.exists():
        cfg_path.unlink()

    out_root = BASE_DIR / "results_stage_f_restriped" / f"w{w}"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest_stage_d()
    manifest_path = cfg_dir / "manifest_stage_d.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    tar_path = cfg_dir / f"worker_input_f{w}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(manifest_path, arcname="manifest_stage_d.json")
        tar.add(BASE_DIR / "code" / "render_tifxyz_112.cpp", arcname="code/render_tifxyz_112.cpp")
        tar.add(BASE_DIR / "code" / "fetch_s3_chunks.py", arcname="code/fetch_s3_chunks.py")
        for t in tasks:
            mg = t["mesh_group"]
            for seg in t["segments"]:
                seg_dir = BASE_DIR / "meshes" / mg / seg
                if seg_dir.exists():
                    tar.add(seg_dir, arcname=f"meshes/{mg}/{seg}")
                else:
                    print(f"WARNING: missing local mesh {seg_dir}", flush=True)

    setup_py = cfg_dir / "remote_setup.py"
    setup_py.write_text(REMOTE_SETUP_SCRIPT)
    seg_runner_py = cfg_dir / "remote_seg_runner.py"
    seg_runner_py.write_text(REMOTE_SEGMENT_RUNNER)

    res = run_colab(cfg_path, ["new", "-s", session_name, "--gpu", args.gpu], check=False)
    if res.returncode != 0 and args.gpu != "L4":
        print(f"A100 unavailable for {session_name}, falling back to L4...", flush=True)
        run_colab(cfg_path, ["new", "-s", session_name, "--gpu", "L4"], check=True)
    elif res.returncode != 0:
        raise RuntimeError(f"Failed to allocate Colab GPU for {session_name}")

    try:
        run_colab(cfg_path, ["upload", "-s", session_name, str(tar_path), "/content/worker_input.tar.gz"])
        run_colab(cfg_path, ["exec", "--timeout", "1800", "-s", session_name, "-f", str(setup_py)])

        for t in tasks:
            scroll = t["scroll"]
            mg = t["mesh_group"]
            orient_mode = t.get("orient_mode", "auto")
            depths = t.get("depths", [-4, 0, 4])
            for seg in t["segments"]:
                done_json = out_root / scroll / seg / "metrics_stage_d.json"
                if done_json.exists():
                    print(f"[skip] {scroll}/{seg} already downloaded locally at {done_json}", flush=True)
                    continue

                job_dict = {
                    "scroll": scroll,
                    "mesh_group": mg,
                    "seg_name": seg,
                    "orient_mode": orient_mode,
                    "depths": depths,
                }
                job_file = cfg_dir / "current_job.json"
                job_file.write_text(json.dumps(job_dict, indent=2))
                run_colab(cfg_path, ["upload", "-s", session_name, str(job_file), "/content/current_job.json"])

                for attempt in range(2):
                    res = run_colab(cfg_path, ["exec", "--timeout", "3600", "-s", session_name, "-f", str(seg_runner_py)], check=False)
                    if res.returncode == 0:
                        break
                    print(f"[warn] attempt {attempt+1} on {scroll}/{seg} returned {res.returncode}, retrying...", flush=True)
                    time.sleep(3)

                local_seg_tar = cfg_dir / f"seg_{scroll}_{seg}.tar.gz"
                if local_seg_tar.exists():
                    local_seg_tar.unlink()
                dl_res = run_colab(cfg_path, ["download", "-s", session_name, "/content/seg_out.tar.gz", str(local_seg_tar)], check=False)
                if dl_res.returncode == 0 and local_seg_tar.exists():
                    with tarfile.open(local_seg_tar, "r:gz") as tf:
                        tf.extractall(path=out_root)
                    print(f"[saved] Unpacked {scroll}/{seg} -> {out_root / scroll / seg}", flush=True)
                else:
                    print(f"[error] Failed to download /content/seg_out.tar.gz for {scroll}/{seg}", flush=True)
    finally:
        run_colab(cfg_path, ["stop", "-s", session_name], check=False)


if __name__ == "__main__":
    main()
