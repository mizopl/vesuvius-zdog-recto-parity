#!/usr/bin/env python3
"""
Stage G — Focused 2.2x2.2 cm Ultra-Dense 1-Voxel Refiner with:
1. True 3D Structure-Tensor Laminar Flattening + 2D Screened-Poisson DCT Integrator
   (up to +-20 voxels normal displacement range!) + 2D Wavefront Sheet-Core Snapping.
2. Ultra-Dense 1-Voxel Normal Depth Sampling:
   13 contiguous depths `d in {-6, -5, -4, -3, -2, -1, 0, +1, +2, +3, +4, +5, +6}` (~9 um step)
   so 2-voxel-thick reed-pen strokes never fall between slices.
3. Dihedral Rotational Test-Time Augmentation (TTA: rot0, rot90, rot180, rot270):
   Eliminates vertical/horizontal fiber directional bias so horizontal crossbars (T, PI, H, E, G)
   and diagonal strokes (A, L, D, N, M, X, Y, K) fire with full strength alongside vertical stems.
4. Continuous 3D Ink-Sheet Ridge Tracking `d*(y, x)` across the 13-depth stack.
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
            "scroll": "PHerc1203",
            "mesh_group": "PHerc1203_upright",
            "seg_name": "auto_grown_20250929222256117_upright",
            "orient": "reverse",
            "target_box_yxwh": [424, 742, 2136, 2136],
        },
        {
            "scroll": "PHerc1203",
            "mesh_group": "PHerc1203_upright",
            "seg_name": "auto_grown_20251005221856743_upright",
            "orient": "reverse",
            "target_box_yxwh": [0, 954, 2136, 2136],
        },
    ],
    1: [
        {
            "scroll": "PHerc0800",
            "mesh_group": "PHerc0800_ext",
            "seg_name": "20251028220955_ext",
            "orient": "reverse",
            "target_box_yxwh": [0, 230, 2315, 2315],
        },
        {
            "scroll": "PHerc0800",
            "mesh_group": "PHerc0800_ext",
            "seg_name": "20251028213516_ext",
            "orient": "reverse",
            "target_box_yxwh": [0, 230, 2315, 2315],
        },
    ],
    2: [
        {
            "scroll": "PHerc0813",
            "mesh_group": "PHerc0813",
            "seg_name": "grid_wrap02_z06457",
            "orient": "reverse",
            "target_box_yxwh": [742, 318, 2136, 2136],
        },
        {
            "scroll": "PHerc1203",
            "mesh_group": "PHerc1203_upright",
            "seg_name": "auto_grown_20250930104534929_upright",
            "orient": "reverse",
            "target_box_yxwh": [636, 106, 2136, 2136],
        },
    ],
    3: [
        {
            "scroll": "PHerc0211",
            "mesh_group": "PHerc0211",
            "seg_name": "grid_wrap00_z09708",
            "orient": "forward",
            "target_box_yxwh": [0, 1060, 2136, 2136],
        },
        {
            "scroll": "PHerc0846A",
            "mesh_group": "PHerc0846A",
            "seg_name": "grid_wrap01_z05327",
            "orient": "reverse",
            "target_box_yxwh": [742, 0, 2136, 2136],
        },
    ],
}

REMOTE_SETUP_SCRIPT = r'''
import json, os, subprocess, sys, tarfile
from pathlib import Path

print("=== STAGE G ULTRA-DENSE 1-VOXEL + TTA + POISSON LAMINAR REFINER SETUP ===", flush=True)
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
from scipy.fft import dctn, idctn
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


def _solve_poisson_tilt_pass(slab_raw: np.ndarray, z_center: float, z_sigma: float, max_tilt: float):
    """Single Screened-Poisson 3D structure-tensor laminar tilt estimation pass on coarse slab."""
    slab = gaussian_filter(slab_raw, sigma=(0.8, 1.35, 1.35))
    Nz, Hc, Wc = slab.shape
    gz = np.zeros_like(slab)
    gy = np.zeros_like(slab)
    gx = np.zeros_like(slab)
    gz[1:-1] = 0.5 * (slab[2:] - slab[:-2])
    gy[:, 1:-1, :] = 0.5 * (slab[:, 2:, :] - slab[:, :-2, :])
    gx[:, :, 1:-1] = 0.5 * (slab[:, :, 2:] - slab[:, :, :-2])

    z_coords = np.arange(Nz, dtype=np.float32)[:, None, None]
    w_z = np.exp(-0.5 * ((z_coords - z_center) / z_sigma) ** 2)
    valid_vox = ((slab_raw > 38.0) & (slab_raw < 245.0)).astype(np.float32)
    w = valid_vox * w_z

    J_zz = gaussian_filter(np.sum(w * gz * gz, axis=0), sigma=4.5)
    J_zx = gaussian_filter(np.sum(w * gz * gx, axis=0), sigma=4.5)
    J_zy = gaussian_filter(np.sum(w * gz * gy, axis=0), sigma=4.5)

    denom = J_zz + np.percentile(J_zz, 25.0) * 0.14 + 1e-4
    p_c = np.clip(-J_zx / denom, -0.50, 0.50)
    q_c = np.clip(-J_zy / denom, -0.50, 0.50)

    div = np.zeros((Hc, Wc), dtype=np.float32)
    div[:, 1:-1] += 0.5 * (p_c[:, 2:] - p_c[:, :-2])
    div[:, 0] += p_c[:, 1] - p_c[:, 0]
    div[:, -1] += p_c[:, -1] - p_c[:, -2]
    div[1:-1, :] += 0.5 * (q_c[2:, :] - q_c[:-2, :])
    div[0, :] += q_c[1, :] - q_c[0, :]
    div[-1, :] += q_c[-1, :] - q_c[-2, :]

    ky = np.arange(Hc, dtype=np.float32)[:, None]
    kx = np.arange(Wc, dtype=np.float32)[None, :]
    eig_y = 2.0 * (1.0 - np.cos(np.pi * ky / float(Hc)))
    eig_x = 2.0 * (1.0 - np.cos(np.pi * kx / float(Wc)))
    lam_screen = (1.0 / 96.0) ** 2
    denom_dct = eig_y + eig_x + lam_screen

    rhs_hat = dctn(-div, type=2, norm="ortho")
    rhs_hat[0, 0] = 0.0
    dz_tilt = idctn(rhs_hat / denom_dct, type=2, norm="ortho").astype(np.float32)
    dz_tilt -= float(np.median(dz_tilt))
    return np.clip(dz_tilt, -max_tilt, max_tilt)


def _warp_coarse_slab_z(slab_raw: np.ndarray, dz_c: np.ndarray) -> np.ndarray:
    """Warp a coarse 3D slab (Nz, Hc, Wc) along z by dz_c(Hc, Wc) with linear interpolation."""
    Nz, Hc, Wc = slab_raw.shape
    yy_c, xx_c = np.mgrid[0:Hc, 0:Wc]
    z_grid = np.arange(Nz, dtype=np.float32)[:, None, None] + dz_c[None, :, :]
    z_clamped = np.clip(z_grid, 0.0, float(Nz - 1.001))
    z0_i = z_clamped.astype(np.int32)
    fz = z_clamped - z0_i
    return (1.0 - fz) * slab_raw[z0_i, yy_c[None, :, :], xx_c[None, :, :]] + fz * slab_raw[z0_i + 1, yy_c[None, :, :], xx_c[None, :, :]]


def poisson_laminar_flatten_roi_cuda(vol112_roi: np.ndarray, max_shift: float = 20.0):
    """
    H1 — Two-Pass Iterative 3D Structure-Tensor Screened-Poisson Laminar Flattener +
    Single-Sheet Core Wavefront Lock on the 2.2x2.2 cm ROI volume `vol112_roi[112, Hr, Wr]`:
    1. Pass 1 estimates large-scale papyrus lamina tilt across z in [25..87] (sigma_z=12.0 vox)
       and solves 2D Screened-Poisson (-laplacian + lambda)*dz_tilt1 = -div(p, q) up to +-16 vox.
    2. Pass 2 warps the coarse slab by dz_tilt1 and runs a tight-band Screened-Poisson refinement
       (sigma_z=6.5 vox around the central lamina) to eliminate residual 1-5 voxel waves.
    3. Locks onto the single continuous bright papyrus sheet core `dz_core(y, x)` in [-7.5, +7.5] vox
       while penalizing dark inter-wrap air gaps (`ct < 48`).
    4. Resamples `vol82_flat[82, Hr, Wr]` on the GPU with sub-voxel linear z-interpolation.
    """
    t0 = time.time()
    Z, Hr, Wr = vol112_roi.shape
    ds = 4
    # Extract 62-layer coarse slab [25..87] centered on z=55.5 in vol112_roi (zc=30.5 in slab_raw)
    slab_raw = vol112_roi[25:87, ::ds, ::ds].astype(np.float32)  # (62, Hc, Wc)
    Nz, Hc, Wc = slab_raw.shape
    zc_mid = 30.5

    # Pass 1: Broad-band Screened-Poisson laminar tilt
    dz_tilt1 = _solve_poisson_tilt_pass(slab_raw, z_center=zc_mid, z_sigma=12.0, max_tilt=16.0)
    slab_warped1 = _warp_coarse_slab_z(slab_raw, dz_tilt1)

    # Pass 2: Tight-band Screened-Poisson laminar refinement around the flattened central sheet
    dz_tilt2 = _solve_poisson_tilt_pass(slab_warped1, z_center=zc_mid, z_sigma=6.5, max_tilt=8.0)
    dz_tilt_c = np.clip(dz_tilt1 + dz_tilt2, -18.0, 18.0)

    # Step 3: Lock onto the single continuous bright papyrus sheet core near zc_mid=30.5
    yy_c, xx_c = np.mgrid[0:Hc, 0:Wc]
    deltas = np.linspace(-7.5, 7.5, 31, dtype=np.float32)  # 0.5-voxel resolution core search
    pot_core = np.zeros((len(deltas), Hc, Wc), dtype=np.float32)

    for idx_d, d_res in enumerate(deltas):
        z_target = np.clip(zc_mid + dz_tilt_c + d_res, 3.0, float(Nz - 4.001))
        z0_i = z_target.astype(np.int32)
        fz = z_target - z0_i
        c_mid = (1.0 - fz) * slab_raw[z0_i, yy_c, xx_c] + fz * slab_raw[z0_i + 1, yy_c, xx_c]
        c_up = slab_raw[np.clip(z0_i - 1, 0, Nz - 1), yy_c, xx_c]
        c_dn = slab_raw[np.clip(z0_i + 2, 0, Nz - 1), yy_c, xx_c]
        core_mean = 0.5 * c_mid + 0.25 * c_up + 0.25 * c_dn
        is_gap = ((core_mean > 0) & (core_mean < 48.0)).astype(np.float32)
        gap_smooth = gaussian_filter(is_gap, sigma=3.0)
        pot_core[idx_d] = gaussian_filter((core_mean - 55.0) / 80.0 - 2.8 * gap_smooth, sigma=6.0) - 0.08 * ((d_res / 6.0) ** 2)

    mid_mask = (deltas >= -4.5) & (deltas <= 4.5)
    global_scores = [float(pot_core[i].mean()) if mid_mask[i] else -1e9 for i in range(len(deltas))]
    best_global_idx = int(np.argmax(global_scores))
    d_global = float(deltas[best_global_idx])

    prior_loc = -0.38 * ((deltas[:, None, None] - d_global) ** 2)
    logits_c = (pot_core + prior_loc) * 5.0
    logits_c -= logits_c.max(axis=0, keepdims=True)
    w_c = np.exp(np.clip(logits_c, -25.0, 0.0))
    w_c /= np.maximum(w_c.sum(axis=0, keepdims=True), 1e-6)
    dz_core_c = gaussian_filter(np.sum(w_c * deltas[:, None, None], axis=0), sigma=8.5)

    # Evaluate residual lamina slope after full flattening to verify H1 (< 2.0 vox residual wave)
    slab_flat_check = _warp_coarse_slab_z(slab_raw, np.clip(dz_tilt_c + dz_core_c, -max_shift, max_shift))
    dz_resid_check = _solve_poisson_tilt_pass(slab_flat_check, z_center=zc_mid, z_sigma=5.5, max_tilt=8.0)
    resid_std_vox = float(dz_resid_check.std())

    dz_total_c = np.clip(dz_tilt_c + dz_core_c, -max_shift, max_shift).astype(np.float32)
    dz_full = cv2.resize(dz_total_c, (Wr, Hr), interpolation=cv2.INTER_CUBIC)
    dz_full = np.clip(dz_full, -max_shift, max_shift)

    # Resample 82-layer volume around z_center (15..96 in vol112) + dz_full(y, x) on GPU
    vol82_out = np.zeros((82, Hr, Wr), dtype=np.uint8)
    strip_h = 512
    z_idx_base = torch.arange(82, device="cuda", dtype=torch.float32)[:, None, None] + 15.0

    with torch.no_grad():
        for y0 in range(0, Hr, strip_h):
            y1 = min(Hr, y0 + strip_h)
            v_strip = torch.from_numpy(vol112_roi[:, y0:y1, :]).to("cuda", dtype=torch.float32)
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

    stats = {
        "dz_tilt_pass1_std": round(float(dz_tilt1.std()), 2),
        "dz_tilt_pass2_std": round(float(dz_tilt2.std()), 2),
        "dz_tilt_min": round(float(dz_tilt_c.min()), 2),
        "dz_tilt_max": round(float(dz_tilt_c.max()), 2),
        "dz_core_global": round(d_global, 2),
        "dz_total_min": round(float(dz_full.min()), 2),
        "dz_total_max": round(float(dz_full.max()), 2),
        "dz_total_std": round(float(dz_full.std()), 2),
        "post_flatten_residual_wave_std_vox": round(resid_std_vox, 3),
        "elapsed_s": round(time.time() - t0, 2),
    }
    print(
        f"  [H1 2-Pass Poisson Laminar Restripe] tilt={stats['dz_tilt_min']:+.1f}..{stats['dz_tilt_max']:+.1f} "
        f"core={stats['dz_core_global']:+.1f} total={stats['dz_total_min']:+.1f}..{stats['dz_total_max']:+.1f} vox "
        f"(applied_std={stats['dz_total_std']:.2f}, post_resid_std={stats['post_flatten_residual_wave_std_vox']:.2f} vox) in {stats['elapsed_s']}s",
        flush=True,
    )
    return vol82_out, dz_full, stats


def eval_villa_single_rot_cuda(vol17_gpu: torch.Tensor, supp_gpu: torch.Tensor):
    """Run dnative and fused seed42+seed43 on an already-on-GPU (17, Vh, Vw) tensor."""
    _, Vh, Vw = vol17_gpu.shape
    supp_np = supp_gpu.cpu().numpy()

    # 1) dnative
    m_dn_model, _, w_dn, p_dn = villa_models["dnative"]
    ys_dn = _sliding_positions_1d(Vh, p_dn, p_dn // 2)
    xs_dn = _sliding_positions_1d(Vw, p_dn, p_dn // 2)
    coords_dn = []
    for y0 in ys_dn:
        vh = min(p_dn, Vh - y0)
        for x0 in xs_dn:
            vw = min(p_dn, Vw - x0)
            if supp_np[y0:y0 + vh, x0:x0 + vw].any():
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

    out_dn = torch.where(wsum_dn > 1e-6, prob_dn / wsum_dn.clamp_min(1e-6), torch.zeros_like(prob_dn)).clamp(0.0, 1.0)
    del prob_dn, wsum_dn

    # 2) seed42 + seed43
    m_42_model, _, w_128, p_128 = villa_models["seed42"]
    m_43_model, _, _, _ = villa_models["seed43"]
    ys_128 = _sliding_positions_1d(Vh, p_128, p_128 // 2)
    xs_128 = _sliding_positions_1d(Vw, p_128, p_128 // 2)
    coords_128 = []
    for y0 in ys_128:
        vh = min(p_128, Vh - y0)
        for x0 in xs_128:
            vw = min(p_128, Vw - x0)
            if supp_np[y0:y0 + vh, x0:x0 + vw].any():
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

    out_42 = torch.where(wsum_128 > 1e-6, prob_42 / wsum_128.clamp_min(1e-6), torch.zeros_like(prob_42)).clamp(0.0, 1.0)
    out_43 = torch.where(wsum_128 > 1e-6, prob_43 / wsum_128.clamp_min(1e-6), torch.zeros_like(prob_43)).clamp(0.0, 1.0)
    del prob_42, prob_43, wsum_128
    return out_dn, out_42, out_43


def eval_villa_tta_cuda(vol17: np.ndarray, supp_2d: np.ndarray, n_rot: int = 4):
    """
    H3 — Run Villa 3-model ensemble with Simultaneous Dihedral Rotational TTA + Tile-Phase Shifting:
    - Each rotation k in {0, 1, 2, 3} is coupled with a distinct spatial tile-phase offset
      (0, 16, 24, 8 px) so 64x64 and 128x128 patch boundary seams never align across passes.
    - Combines via 65% mean + 35% directional max so horizontal crossbars (T, PI, H, E, G)
      and diagonal strokes (A, L, D, N, M, X, Y, K) connect cleanly with vertical stems.
    """
    t0 = time.time()
    vol17_base = torch.from_numpy(vol17).to("cuda", non_blocking=True)
    supp_base = torch.from_numpy(supp_2d).to("cuda", non_blocking=True)

    rots = [0, 1, 2, 3] if n_rot == 4 else [0, 1]
    phase_pads = {0: 0, 1: 16, 2: 24, 3: 8}
    dn_list, s42_list, s43_list = [], [], []
    for k in rots:
        v_rot = torch.rot90(vol17_base, k=k, dims=(1, 2)).contiguous()
        s_rot = torch.rot90(supp_base, k=k, dims=(0, 1)).contiguous()
        pad_px = phase_pads.get(k, 0)
        if pad_px > 0:
            v_in = torch.nn.functional.pad(v_rot, (pad_px, 0, pad_px, 0), mode="constant", value=0)
            s_in = torch.nn.functional.pad(s_rot, (pad_px, 0, pad_px, 0), mode="constant", value=False)
            odn, o42, o43 = eval_villa_single_rot_cuda(v_in, s_in)
            odn = odn[pad_px:, pad_px:]
            o42 = o42[pad_px:, pad_px:]
            o43 = o43[pad_px:, pad_px:]
        else:
            odn, o42, o43 = eval_villa_single_rot_cuda(v_rot, s_rot)
        dn_list.append(torch.rot90(odn, k=(4 - k) % 4, dims=(0, 1)))
        s42_list.append(torch.rot90(o42, k=(4 - k) % 4, dims=(0, 1)))
        s43_list.append(torch.rot90(o43, k=(4 - k) % 4, dims=(0, 1)))

    def fuse_tta(t_list):
        st = torch.stack(t_list, dim=0)
        blended = 0.65 * st.mean(dim=0) + 0.35 * st.max(dim=0).values
        return (blended.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()

    m_dn_u8 = fuse_tta(dn_list)
    m_42_u8 = fuse_tta(s42_list)
    m_43_u8 = fuse_tta(s43_list)
    del vol17_base, supp_base, dn_list, s42_list, s43_list
    torch.cuda.empty_cache()
    return m_dn_u8, m_42_u8, m_43_u8, time.time() - t0


def eval_hecate_single_rot_cuda(sub48_gpu: torch.Tensor, supp_gpu: torch.Tensor, bs: int = 48):
    d, H, W = sub48_gpu.shape
    p = hecate_p
    stride = p // 2
    supp_np = supp_gpu.cpu().numpy()

    ys = _sliding_positions_1d(H, p, stride)
    xs = _sliding_positions_1d(W, p, stride)
    coords = []
    for y0 in ys:
        ph = min(p, H - y0)
        for x0 in xs:
            pw = min(p, W - x0)
            if supp_np[y0:y0 + ph, x0:x0 + pw].any():
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

    out = torch.where(weight_sum > 1e-6, prob_sum / weight_sum.clamp_min(1e-6), torch.zeros_like(prob_sum)).clamp(0.0, 1.0)
    del prob_sum, weight_sum
    return out


def eval_hecate_tta_cuda(vol66: np.ndarray, supp_2d: np.ndarray, reverse: bool, n_rot: int = 1):
    t0 = time.time()
    depth, H, W = vol66.shape
    d = hecate_d
    z0 = depth // 2 - d // 2
    zs = slice(depth - z0 - d, depth - z0) if reverse else slice(z0, z0 + d)
    sub48 = vol66[zs][::-1].copy() if reverse else vol66[zs].copy()

    sub48_base = torch.from_numpy(sub48).to("cuda", non_blocking=True)
    supp_base = torch.from_numpy(supp_2d).to("cuda", non_blocking=True)

    rots = [0, 1] if n_rot == 2 else [0]
    hc_list = []
    for k in rots:
        v_rot = torch.rot90(sub48_base, k=k, dims=(1, 2)).contiguous()
        s_rot = torch.rot90(supp_base, k=k, dims=(0, 1)).contiguous()
        if k == 1:
            pad_px = 16
            v_in = torch.nn.functional.pad(v_rot, (pad_px, 0, pad_px, 0), mode="constant", value=0)
            s_in = torch.nn.functional.pad(s_rot, (pad_px, 0, pad_px, 0), mode="constant", value=False)
            ohc = eval_hecate_single_rot_cuda(v_in, s_in)[pad_px:, pad_px:]
        else:
            ohc = eval_hecate_single_rot_cuda(v_rot, s_rot)
        hc_list.append(torch.rot90(ohc, k=(4 - k) % 4, dims=(0, 1)))

    if len(hc_list) == 2:
        blended = 0.65 * (hc_list[0] + hc_list[1]) * 0.5 + 0.35 * torch.maximum(hc_list[0], hc_list[1])
    else:
        blended = hc_list[0]

    m_hc_u8 = torch.round(blended.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    del sub48_base, supp_base, hc_list, blended
    torch.cuda.empty_cache()
    return m_hc_u8, time.time() - t0


def render_ens_and_gold_from_z(ct_slice: np.ndarray, ens_z: np.ndarray, valid_mask: np.ndarray):
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
    ct_f = ct_slice.astype(np.float32)[..., None]
    gold = np.array([255.0, 235.0, 120.0], dtype=np.float32)[None, None, :]
    rgb = np.clip(ct_f * 0.45 * (1.0 - alpha[..., None]) + gold * alpha[..., None], 0.0, 255.0).astype(np.uint8)
    return ens_u8, rgb


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
    ens_u8, rgb = render_ens_and_gold_from_z(ct_mid, ens_z, valid_mask)
    return ens_z, ens_u8, rgb


manifest = json.load(open("/content/manifest_stage_d.json"))
job = json.load(open("/content/current_job.json"))
scroll = job["scroll"]
mesh_group = job["mesh_group"]
seg_name = job["seg_name"]
orient = job["orient"]
target_box = job["target_box_yxwh"]  # [y0_2cm, x0_2cm, wh_2cm, ww_2cm]

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
print(f"=== Starting Stage G Ultra-Dense 1-Voxel Refiner on {scroll}/{seg_name} ({orient}) ===", flush=True)
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

# Extract focused 2.2 x 2.2 cm ROI (~1.0 mm padding around the 2.0 x 2.0 cm prize window)
ty0, tx0, twh, tww = target_box
pad_px = int(round(1000.0 / um))  # 1.0 mm margin on each side
ry0 = max(0, ty0 - pad_px)
rx0 = max(0, tx0 - pad_px)
ry1 = min(H, ty0 + twh + pad_px)
rx1 = min(W, tx0 + tww + pad_px)
# Inner 2.0 x 2.0 cm window coordinates inside the ROI
iy0 = ty0 - ry0
ix0 = tx0 - rx0
iwh = min(twh, ry1 - ry0 - iy0)
iww = min(tww, rx1 - rx0 - ix0)

vol112_roi = vol112[:, ry0:ry1, rx0:rx1].copy()
del vol112
gc.collect()
Hr, Wr = vol112_roi.shape[1], vol112_roi.shape[2]
print(f"  Extracted 112-layer focused ROI ({Hr}x{Wr} px, inner 2x2cm=[{iy0},{ix0},{iwh},{iww}]) in {time.time()-t_seg:.1f}s", flush=True)

out_dir = Path(f"/content/results/{scroll}/{seg_name}")
out_dir.mkdir(parents=True, exist_ok=True)

# Save raw un-flattened central CT slice for before/after comparison
cv2.imwrite(str(out_dir / "ct_mid_raw_unflattened.png"), vol112_roi[55])

# H1: Run 2-Pass Iterative 3D Structure-Tensor Screened-Poisson Laminar Flattener + Single-Sheet Core Lock!
vol82_roi, dz_full, restripe_stats = poisson_laminar_flatten_roi_cuda(vol112_roi, max_shift=20.0)
del vol112_roi
gc.collect()

dz_vis = np.clip((dz_full + 20.0) / 40.0 * 255.0, 0.0, 255.0).astype(np.uint8)
cv2.imwrite(str(out_dir / "delta_z_poisson_map.png"), dz_vis)
cv2.imwrite(str(out_dir / "ct_mid.png"), vol82_roi[42])

valid_mask = (vol82_roi[42] > 0)
v_inner = valid_mask[iy0:iy0+iwh, ix0:ix0+iww]
rev_flag = (orient == "reverse")

# Ultra-dense 1-voxel normal depth step: 13 slices d in {-6, -5, -4, -3, -2, -1, 0, +1, +2, +3, +4, +5, +6}
depths = list(range(-6, 7))
depth_results = {}
dn_stack, s42_stack, s43_stack, hc_stack, ct_stack = [], [], [], [], []

for d in depths:
    t_d = time.time()
    s0 = 8 + d
    vol_d = vol82_roi[s0:s0 + 66]
    z0_v = 26 if rev_flag else 23
    sub17 = vol_d[z0_v:z0_v + 17]
    sub17 = sub17[::-1].copy() if rev_flag else sub17.copy()
    supp_2d = sub17.any(axis=0)

    # Use 4x Dihedral + Tile-Phase TTA on core depths {-3..+3} and 2x TTA (rot0 + rot90) on outer depths
    v_nrot = 4 if abs(d) <= 3 else 2
    h_nrot = 2 if d in (-2, -1, 0, 1, 2) else 1

    m_dn, m_42, m_43, t_v = eval_villa_tta_cuda(sub17, supp_2d, n_rot=v_nrot)
    m_hc, t_h = eval_hecate_tta_cuda(vol_d, supp_2d, reverse=rev_flag, n_rot=h_nrot)

    ct_slice = vol_d[34].copy()
    villa_f = (m_dn.astype(np.float32) + m_42.astype(np.float32) + m_43.astype(np.float32)) / 3.0
    hecate_f = m_hc.astype(np.float32)
    v_v, v_h = villa_f[valid_mask], hecate_f[valid_mask]
    r_vh = float(np.corrcoef(v_v, v_h)[0, 1]) if (v_v.size > 500 and v_v.std() > 1e-3 and v_h.std() > 1e-3) else 0.0
    v_vi = villa_f[iy0:iy0+iwh, ix0:ix0+iww][v_inner]
    v_hi = hecate_f[iy0:iy0+iwh, ix0:ix0+iww][v_inner]
    r_vh_2x2 = float(np.corrcoef(v_vi, v_hi)[0, 1]) if (v_vi.size > 500 and v_vi.std() > 1e-3 and v_hi.std() > 1e-3) else r_vh

    ens4 = 0.25 * (m_dn.astype(np.float32) + m_42.astype(np.float32) + m_43.astype(np.float32) + m_hc.astype(np.float32))
    p50, p995 = [float(x) for x in np.percentile(ens4[valid_mask], [50.0, 99.5])] if v_v.size > 500 else (0.0, 0.0)
    hc_p995 = float(np.percentile(v_h, 99.5)) if v_h.size > 500 else 0.0

    d_key = f"d{d:+d}"
    depth_results[d_key] = {
        "ens4_p50": round(p50, 2),
        "ens4_p995": round(p995, 2),
        "hecate_p995": round(hc_p995, 2),
        "r_villa_hecate": round(r_vh, 4),
        "r_2x2cm": round(r_vh_2x2, 4),
        "villa_tta_rots": v_nrot,
        "hecate_tta_rots": h_nrot,
    }
    cv2.imwrite(str(out_dir / f"dnative_{orient}_{d_key}.png"), m_dn)
    cv2.imwrite(str(out_dir / f"hecate_{orient}_{d_key}.png"), m_hc)

    dn_stack.append(m_dn.astype(np.float32))
    s42_stack.append(m_42.astype(np.float32))
    s43_stack.append(m_43.astype(np.float32))
    hc_stack.append(m_hc.astype(np.float32))
    ct_stack.append(ct_slice.astype(np.float32))
    print(f"  [{scroll}/{seg_name} d={d:+2d} (V:{v_nrot}x,H:{h_nrot}x)] r_roi={r_vh:.4f} r_2x2cm={r_vh_2x2:.4f} ens4_p99.5={p995:.1f} hc_p99.5={hc_p995:.1f} ({time.time()-t_d:.1f}s)", flush=True)

# H2: Global 13-Depth Stack Z-Score Normalization (Nieuwlaar ensemble_maps.py parity!)
dn_arr = np.stack(dn_stack, axis=0)   # (13, Hr, Wr)
s42_arr = np.stack(s42_stack, axis=0)
s43_arr = np.stack(s43_stack, axis=0)
hc_arr = np.stack(hc_stack, axis=0)
ct_arr = np.stack(ct_stack, axis=0)

def global_znorm_3d(arr3d: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    vals = arr3d[:, mask2d]
    if vals.size < 100:
        return np.zeros_like(arr3d)
    mu = float(vals.mean())
    sd = max(float(vals.std()), 1e-4)
    return (arr3d - mu) / sd

ens_z_global_3d = 0.25 * (
    global_znorm_3d(dn_arr, valid_mask)
    + global_znorm_3d(s42_arr, valid_mask)
    + global_znorm_3d(s43_arr, valid_mask)
    + global_znorm_3d(hc_arr, valid_mask)
)

# Save global-z-normalized per-depth maps and record global-z p99.5 per depth for H4 Attenuation Profile
for idx_d, d in enumerate(depths):
    d_key = f"d{d:+d}"
    z_slice = ens_z_global_3d[idx_d]
    z_u8, _ = render_ens_and_gold_from_z(ct_arr[idx_d], z_slice, valid_mask)
    cv2.imwrite(str(out_dir / f"ens_z_{orient}_{d_key}.png"), z_u8)
    z_p995 = float(np.percentile(z_slice[valid_mask], 99.5)) if valid_mask.sum() > 500 else 0.0
    depth_results[d_key]["global_z_p995"] = round(z_p995, 4)

# H2a: Select optimal Single Constant Depth Slice d* in [-4..+4] (Nieuwlaar Single-Layer Output)
single_scores = []
for idx_d, d in enumerate(depths):
    d_key = f"d{d:+d}"
    if abs(d) <= 4:
        sc = depth_results[d_key]["r_2x2cm"] + 0.04 * depth_results[d_key]["global_z_p995"]
    else:
        sc = -1e9
    single_scores.append(sc)
i_best = int(np.argmax(single_scores))
d_best_val = float(depths[i_best])
best_d_key = f"d{int(d_best_val):+d}"

ct_single = ct_arr[i_best].astype(np.uint8)
ens_z_single_u8, gold_single_rgb = render_ens_and_gold_from_z(ct_single, ens_z_global_3d[i_best], valid_mask)
cv2.imwrite(str(out_dir / "ct_mid_single_dstar.png"), ct_single)
cv2.imwrite(str(out_dir / f"ens_z_{orient}_single_dstar.png"), ens_z_single_u8)
cv2.imwrite(str(out_dir / f"submission_gold_{orient}_single_dstar.png"), cv2.cvtColor(gold_single_rgb, cv2.COLOR_RGB2BGR))

# H2b: Continuous 1-Voxel Ink-Sheet Ridge Tracking around d_best_val (tight +-2.5 vox band, sigma_d=1.15 vox)
r_list = np.array([max(0.02, depth_results[f"d{d:+d}"]["r_2x2cm"]) for d in depths], dtype=np.float32)
stroke_energy = np.stack(
    [gaussian_filter(np.maximum(ens_z_global_3d[i], 0.0), sigma=28.0) * (r_list[i] ** 0.85) for i in range(len(depths))],
    axis=0,
)
d_grid = np.array(depths, dtype=np.float32)[:, None, None]
prior_d = -0.35 * ((d_grid - d_best_val) ** 2)
logits_ridge = 3.5 * stroke_energy + prior_d
logits_ridge -= logits_ridge.max(axis=0, keepdims=True)
w_ridge_raw = np.exp(np.clip(logits_ridge, -20.0, 0.0))
w_ridge_raw /= np.maximum(w_ridge_raw.sum(axis=0, keepdims=True), 1e-6)
d_star = gaussian_filter(np.sum(w_ridge_raw * d_grid, axis=0), sigma=52.0)

# Narrow 1.15-voxel Gaussian slab centered on d*(y, x) so strokes connect across 1-voxel steps without multi-wrap bleed
w_slab = np.exp(-0.5 * ((d_grid - d_star[None, :, :]) / 1.15) ** 2)
w_slab /= np.maximum(w_slab.sum(axis=0, keepdims=True), 1e-6)

dn_md = (w_slab * dn_arr).sum(axis=0).astype(np.uint8)
s42_md = (w_slab * s42_arr).sum(axis=0).astype(np.uint8)
s43_md = (w_slab * s43_arr).sum(axis=0).astype(np.uint8)
hc_md = (w_slab * hc_arr).sum(axis=0).astype(np.uint8)
ct_md = (w_slab * ct_arr).sum(axis=0).astype(np.uint8)
ens_z_md_f = (w_slab * ens_z_global_3d).sum(axis=0)
ens_z_md_u8, gold_md_rgb = render_ens_and_gold_from_z(ct_md, ens_z_md_f, valid_mask)

villa_md_f = (dn_md.astype(np.float32) + s42_md.astype(np.float32) + s43_md.astype(np.float32)) / 3.0
hecate_md_f = hc_md.astype(np.float32)
v_v, v_h = villa_md_f[valid_mask], hecate_md_f[valid_mask]
r_md = float(np.corrcoef(v_v, v_h)[0, 1]) if (v_v.size > 500 and v_v.std() > 1e-3 and v_h.std() > 1e-3) else 0.0
r_2x2cm = float(np.corrcoef(villa_md_f[iy0:iy0+iwh, ix0:ix0+iww][v_inner], hecate_md_f[iy0:iy0+iwh, ix0:ix0+iww][v_inner])[0, 1]) if v_inner.sum() > 500 else r_md
ens4_md = 0.25 * (dn_md.astype(np.float32) + s42_md.astype(np.float32) + s43_md.astype(np.float32) + hc_md.astype(np.float32))
p50_md, p995_md = [float(x) for x in np.percentile(ens4_md[valid_mask], [50.0, 99.5])] if v_v.size > 500 else (0.0, 0.0)
hc_p995_md = float(np.percentile(v_h, 99.5)) if v_h.size > 500 else 0.0

# Compute H4 Attenuation Profile ratio: global_z_p995 at d* vs d*+-4 voxels
idx_m4 = max(0, i_best - 4)
idx_p4 = min(len(depths) - 1, i_best + 4)
z_peak = max(depth_results[best_d_key]["global_z_p995"], 1e-3)
z_off4 = 0.5 * (depth_results[f"d{depths[idx_m4]:+d}"]["global_z_p995"] + depth_results[f"d{depths[idx_p4]:+d}"]["global_z_p995"])
attenuation_drop_pct = round(float((z_peak - z_off4) / max(abs(z_peak), 1.0) * 100.0), 1)

cv2.imwrite(str(out_dir / "ct_mid_ridge.png"), ct_md)
cv2.imwrite(str(out_dir / f"ens_z_{orient}_multidepth.png"), ens_z_md_u8)
cv2.imwrite(str(out_dir / f"submission_gold_{orient}_multidepth.png"), cv2.cvtColor(gold_md_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(out_dir / f"dnative_{orient}_multidepth.png"), dn_md)
cv2.imwrite(str(out_dir / f"hecate_{orient}_multidepth.png"), hc_md)
cv2.imwrite(str(out_dir / f"s42_{orient}_multidepth.png"), s42_md)
cv2.imwrite(str(out_dir / f"s43_{orient}_multidepth.png"), s43_md)

d_star_vis = np.clip((d_star + 6.0) / 12.0 * 255.0, 0.0, 255.0).astype(np.uint8)
cv2.imwrite(str(out_dir / "d_star_ink_ridge_map.png"), d_star_vis)

seg_report = {
    "scroll": scroll,
    "segment": seg_name,
    "um": um,
    "full_shape_hw": [H, W],
    "roi_yxwh": [int(ry0), int(rx0), int(Hr), int(Wr)],
    "inner_2x2cm_yxwh": [int(iy0), int(ix0), int(iwh), int(iww)],
    "chosen_orient": orient,
    "restripe_stats": restripe_stats,
    "best_single_depth": best_d_key,
    "single_dstar_metrics": depth_results[best_d_key],
    "h4_attenuation_drop_at_pm4_pct": attenuation_drop_pct,
    "d_star_ridge_std_vox": round(float(d_star[valid_mask].std()), 3),
    "depths": depth_results,
    "multidepth": {
        "ens4_p50": round(p50_md, 2),
        "ens4_p995": round(p995_md, 2),
        "ens4_contrast": round(p995_md - p50_md, 2),
        "hecate_p995": round(hc_p995_md, 2),
        "r_villa_hecate": round(r_md, 4),
        "max_2x2cm_r_villa_hecate": round(r_2x2cm, 4),
        "max_2x2cm_box_yxwh": [int(iy0), int(ix0), int(iwh), int(iww)],
    },
    "elapsed_s": round(time.time() - t_seg, 1),
}
with open(out_dir / "metrics_stage_g.json", "w") as f:
    json.dump(seg_report, f, indent=2)

print(
    f"  ==> [{scroll}/{seg_name} STAGE G H1-H3] best_d*={best_d_key} (r_2x2_d*={depth_results[best_d_key]['r_2x2cm']:.4f}) | "
    f"ridge: r_roi={r_md:.4f} r_2x2cm={r_2x2cm:.4f} ens4_p99.5={p995_md:.1f} hc_p99.5={hc_p995_md:.1f} "
    f"d*_std={seg_report['d_star_ridge_std_vox']:.2f}vox ({seg_report['elapsed_s']}s total)",
    flush=True,
)
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
    session_name = f"vesuvius-g{w}"
    cfg_dir = Path(f"/tmp/colab_g{w}")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "sessions.json"
    if cfg_path.exists():
        cfg_path.unlink()

    out_root = BASE_DIR / "results_stage_g" / f"w{w}"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest_stage_d()
    manifest_path = cfg_dir / "manifest_stage_d.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    tar_path = cfg_dir / f"worker_input_g{w}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(manifest_path, arcname="manifest_stage_d.json")
        tar.add(BASE_DIR / "code" / "render_tifxyz_112.cpp", arcname="code/render_tifxyz_112.cpp")
        tar.add(BASE_DIR / "code" / "fetch_s3_chunks.py", arcname="code/fetch_s3_chunks.py")
        added = set()
        for t in tasks:
            mg = t["mesh_group"]
            seg = t["seg_name"]
            if (mg, seg) in added:
                continue
            added.add((mg, seg))
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
            seg = t["seg_name"]
            done_json = out_root / scroll / seg / "metrics_stage_g.json"
            if done_json.exists():
                print(f"[skip] {scroll}/{seg} already downloaded locally at {done_json}", flush=True)
                continue

            job_file = cfg_dir / "current_job.json"
            job_file.write_text(json.dumps(t, indent=2))
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
                print(f"[saved] Unpacked Stage G {scroll}/{seg} -> {out_root / scroll / seg}", flush=True)
            else:
                print(f"[error] Failed to download /content/seg_out.tar.gz for {scroll}/{seg}", flush=True)
    finally:
        run_colab(cfg_path, ["stop", "-s", session_name], check=False)


if __name__ == "__main__":
    main()
