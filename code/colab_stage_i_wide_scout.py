#!/usr/bin/env python3
"""
Stage I — Fast Wide-Area m7-Snap Column Scout + Kernel-Matched Z-DoG (`I_thin`) + Auto-Promoter
Implements the hybrid architecture `H_E = H_D + H_A + H_B + H_C + H_F + H_H`:

1. Loads 3D `.tifxyz` candidate windows (`116x116` vertices = `2320x2320 px` ~ `2.15x2.15 cm`)
   already snapped in 3D to S3 `*-surface-m7-L0-th0.2.zarr/0` (or `*-m7-L2-th0.2.zarr/0` for
   the 2026-03 `2.403 um / 77 keV` synchrotron zoom scans).
2. Renders the 82-layer pure 3D normal slab (`render_tifxyz_82`, ZERO post-render 2D Poisson warping).
3. Computes CPU/CT Stratigraphic & Physical Diagnostics (`H_D`, `H_F`, `H_H`):
   - Bandpass Fiber Anisotropy `A_RV(s) = E[(d_y I_bp)^2] / E[(d_x I_bp)^2]` across the lamina,
   - Dual-Face Trans-Lamellar Differential `delta_A_RV = A_RV(s_inward) - A_RV(s_outward)` (`H_H`),
   - Horizontal Line-Row Periodicity (`row_periodicity_score` at 3.2-4.8 mm pitch, `H_F`).
4. Runs Two-Stage Fast GPU Scout (`dense_native + Hecate` at `1x TTA bf16`):
   - Stage 4A (`d = 0`): Tests `reverse` vs `forward` orientation (~11s total) and picks `orient_best`.
   - Stage 4B (`d in {-6, -4, -2, 0, +2, +4, +6}`): Evaluates the 7-point 2-voxel stack so that ANY
     best surface depth `d* in {-2, 0, +2}` (including the inward *recto* wall) has exact symmetric
     off-lamina reference layers `d* - 4` and `d* + 4`!
5. Applies Kernel-Matched Thin-Sheet Surface Derivative (`I_thin`, `H_D` / `Bet 6`):
   - `I_thin_dn(d*) = max(0, I_dn(d*) - 0.75 * 0.5 * (I_dn(d*-4) + I_dn(d*+4)))`
   - `I_thin_hc(d*) = max(0, I_hc(d*) - 0.90 * 0.5 * (I_hc(d*-4) + I_hc(d*+4)))`
   - Scale-Locked 3D Stack Normalization (eliminates per-layer `[1.0, 99.9]` stretching bug from Stage H).
   - Frozen-Stroke-Mask Attenuation (`stroke_atten_pm4_pct` on `M_stroke = {I(d*) >= P90}`).
6. In-Session Auto-Promotion (`Bet 7` & `Bet 8`):
   - If a candidate passes Gate 1 (`r_2x2cm >= 0.24`) AND Gate 2 (`stroke_atten_pm4_pct >= 32.0%`),
     immediately promotes it in RAM (`4x TTA` + `seed42 + seed43` across `d* - 4, d* - 2, d* - 1, d*, d* + 1, d* + 2, d* + 4`).
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

REMOTE_SETUP_SCRIPT = r'''
import json, os, subprocess, sys, tarfile
from pathlib import Path

print("=== STAGE I WIDE-AREA M7-SNAP COLUMN SCOUT + THIN-LAYER Z-DOG SETUP ===", flush=True)
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
    "/content/code/render_tifxyz_82.cpp", "-o", "/content/code/render_tifxyz_82"
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

REMOTE_SCOUT_RUNNER = r'''
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
    print(f"Loaded 4 models on {GPU_NAME} in {time.time()-t_load:.1f}s", flush=True)


def eval_dnative_only_cuda(vol17: np.ndarray, supp_2d: np.ndarray, n_rot: int = 1):
    """Fast dnative-only evaluation (~1.8s on A100 for 2320x2320 at 1x TTA)."""
    vol17_base = torch.from_numpy(vol17).to("cuda", non_blocking=True)
    supp_base = torch.from_numpy(supp_2d).to("cuda", non_blocking=True)
    m_dn_model, _, w_dn, p_dn = villa_models["dnative"]
    rots = [0, 1, 2, 3][:n_rot]
    phase_pads = {0: 0, 1: 16, 2: 24, 3: 8}
    dn_list = []

    for k in rots:
        v_rot = torch.rot90(vol17_base, k=k, dims=(1, 2)).contiguous()
        s_rot = torch.rot90(supp_base, k=k, dims=(0, 1)).contiguous()
        pad_px = phase_pads.get(k, 0)
        if pad_px > 0:
            v_in = torch.nn.functional.pad(v_rot, (pad_px, 0, pad_px, 0), mode="constant", value=0)
            s_in = torch.nn.functional.pad(s_rot, (pad_px, 0, pad_px, 0), mode="constant", value=False)
        else:
            v_in, s_in = v_rot, s_rot

        _, Vh, Vw = v_in.shape
        supp_np = s_in.cpu().numpy()
        ys_dn = _sliding_positions_1d(Vh, p_dn, p_dn // 2)
        xs_dn = _sliding_positions_1d(Vw, p_dn, p_dn // 2)
        coords_dn = [
            (y0, x0, min(p_dn, Vh - y0), min(p_dn, Vw - x0))
            for y0 in ys_dn for x0 in xs_dn
            if supp_np[y0:y0 + min(p_dn, Vh - y0), x0:x0 + min(p_dn, Vw - x0)].any()
        ]
        prob_dn = torch.zeros((Vh, Vw), dtype=torch.float32, device="cuda")
        wsum_dn = torch.zeros((Vh, Vw), dtype=torch.float32, device="cuda")
        bs_dn = 96 if IS_A100 else 64
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for start in range(0, len(coords_dn), bs_dn):
                bc = coords_dn[start:start + bs_dn]
                B = len(bc)
                patches = torch.zeros((B, 17, p_dn, p_dn), dtype=torch.float32, device="cuda")
                supps = torch.zeros((B, p_dn, p_dn), dtype=torch.float32, device="cuda")
                for bi, (y0, x0, vh, vw) in enumerate(bc):
                    sub = v_in[:, y0:y0 + vh, x0:x0 + vw].to(torch.float32)
                    s = s_in[y0:y0 + vh, x0:x0 + vw]
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
        odn = torch.where(wsum_dn > 1e-6, prob_dn / wsum_dn.clamp_min(1e-6), torch.zeros_like(prob_dn)).clamp(0.0, 1.0)
        if pad_px > 0:
            odn = odn[pad_px:, pad_px:]
        dn_list.append(torch.rot90(odn, k=(4 - k) % 4, dims=(0, 1)))
        del prob_dn, wsum_dn

    if len(dn_list) == 1:
        res = (dn_list[0].clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    else:
        st = torch.stack(dn_list, dim=0)
        blended = 0.65 * st.mean(dim=0) + 0.35 * st.max(dim=0).values
        res = (blended.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    del vol17_base, supp_base, dn_list
    return res


def eval_seed42_43_cuda(vol17: np.ndarray, supp_2d: np.ndarray, n_rot: int = 2):
    """Evaluates seed42 + seed43 when a candidate is promoted."""
    vol17_base = torch.from_numpy(vol17).to("cuda", non_blocking=True)
    supp_base = torch.from_numpy(supp_2d).to("cuda", non_blocking=True)
    m_42_model, _, w_128, p_128 = villa_models["seed42"]
    m_43_model, _, _, _ = villa_models["seed43"]
    rots = [0, 1, 2, 3][:n_rot]
    phase_pads = {0: 0, 1: 16, 2: 24, 3: 8}
    s42_list, s43_list = [], []

    for k in rots:
        v_rot = torch.rot90(vol17_base, k=k, dims=(1, 2)).contiguous()
        s_rot = torch.rot90(supp_base, k=k, dims=(0, 1)).contiguous()
        pad_px = phase_pads.get(k, 0)
        if pad_px > 0:
            v_in = torch.nn.functional.pad(v_rot, (pad_px, 0, pad_px, 0), mode="constant", value=0)
            s_in = torch.nn.functional.pad(s_rot, (pad_px, 0, pad_px, 0), mode="constant", value=False)
        else:
            v_in, s_in = v_rot, s_rot

        _, Vh, Vw = v_in.shape
        supp_np = s_in.cpu().numpy()
        ys_128 = _sliding_positions_1d(Vh, p_128, p_128 // 2)
        xs_128 = _sliding_positions_1d(Vw, p_128, p_128 // 2)
        coords_128 = [
            (y0, x0, min(p_128, Vh - y0), min(p_128, Vw - x0))
            for y0 in ys_128 for x0 in xs_128
            if supp_np[y0:y0 + min(p_128, Vh - y0), x0:x0 + min(p_128, Vw - x0)].any()
        ]
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
                    sub = v_in[:, y0:y0 + vh, x0:x0 + vw].to(torch.float32)
                    s = s_in[y0:y0 + vh, x0:x0 + vw]
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
        o42 = torch.where(wsum_128 > 1e-6, prob_42 / wsum_128.clamp_min(1e-6), torch.zeros_like(prob_42)).clamp(0.0, 1.0)
        o43 = torch.where(wsum_128 > 1e-6, prob_43 / wsum_128.clamp_min(1e-6), torch.zeros_like(prob_43)).clamp(0.0, 1.0)
        if pad_px > 0:
            o42 = o42[pad_px:, pad_px:]
            o43 = o43[pad_px:, pad_px:]
        s42_list.append(torch.rot90(o42, k=(4 - k) % 4, dims=(0, 1)))
        s43_list.append(torch.rot90(o43, k=(4 - k) % 4, dims=(0, 1)))
        del prob_42, prob_43, wsum_128

    def _fuse(t_list):
        if len(t_list) == 1:
            return (t_list[0].clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
        st = torch.stack(t_list, dim=0)
        blended = 0.65 * st.mean(dim=0) + 0.35 * st.max(dim=0).values
        return (blended.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()

    m42 = _fuse(s42_list)
    m43 = _fuse(s43_list)
    del vol17_base, supp_base, s42_list, s43_list
    return m42, m43


def eval_hecate_tta_cuda(vol66: np.ndarray, supp_2d: np.ndarray, reverse: bool, n_rot: int = 1):
    depth, H, W = vol66.shape
    d = hecate_d
    z0 = depth // 2 - d // 2
    zs = slice(depth - z0 - d, depth - z0) if reverse else slice(z0, z0 + d)
    sub48 = vol66[zs][::-1].copy() if reverse else vol66[zs].copy()

    sub48_base = torch.from_numpy(sub48).to("cuda", non_blocking=True)
    supp_base = torch.from_numpy(supp_2d).to("cuda", non_blocking=True)

    rots = [0, 1] if n_rot >= 2 else [0]
    hc_list = []
    p = hecate_p
    stride = p // 2
    divisor = float(hecate_model.divisor)
    bs = 64 if IS_A100 else 48

    for k in rots:
        v_rot = torch.rot90(sub48_base, k=k, dims=(1, 2)).contiguous()
        s_rot = torch.rot90(supp_base, k=k, dims=(0, 1)).contiguous()
        pad_px = 16 if k == 1 else 0
        if pad_px > 0:
            v_in = torch.nn.functional.pad(v_rot, (pad_px, 0, pad_px, 0), mode="constant", value=0)
            s_in = torch.nn.functional.pad(s_rot, (pad_px, 0, pad_px, 0), mode="constant", value=False)
        else:
            v_in, s_in = v_rot, s_rot

        _, Hr, Wr = v_in.shape
        supp_np = s_in.cpu().numpy()
        ys = _sliding_positions_1d(Hr, p, stride)
        xs = _sliding_positions_1d(Wr, p, stride)
        coords = [
            (y0, x0, min(p, Hr - y0), min(p, Wr - x0))
            for y0 in ys for x0 in xs
            if supp_np[y0:y0 + min(p, Hr - y0), x0:x0 + min(p, Wr - x0)].any()
        ]
        prob_sum = torch.zeros((Hr, Wr), dtype=torch.float32, device="cuda")
        weight_sum = torch.zeros((Hr, Wr), dtype=torch.float32, device="cuda")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for start in range(0, len(coords), bs):
                bc = coords[start:start + bs]
                B = len(bc)
                patches = torch.zeros((B, d, p, p), dtype=torch.float32, device="cuda")
                valid_batch = torch.zeros((B, p, p), dtype=torch.bool, device="cuda")
                for bi, (y0, x0, ph, pw) in enumerate(bc):
                    raw = v_in[:, y0:y0 + ph, x0:x0 + pw].to(torch.float32)
                    s = s_in[y0:y0 + ph, x0:x0 + pw]
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
        ohc = torch.where(weight_sum > 1e-6, prob_sum / weight_sum.clamp_min(1e-6), torch.zeros_like(prob_sum)).clamp(0.0, 1.0)
        if pad_px > 0:
            ohc = ohc[pad_px:, pad_px:]
        hc_list.append(torch.rot90(ohc, k=(4 - k) % 4, dims=(0, 1)))
        del prob_sum, weight_sum

    if len(hc_list) == 2:
        blended = 0.65 * (hc_list[0] + hc_list[1]) * 0.5 + 0.35 * torch.maximum(hc_list[0], hc_list[1])
    else:
        blended = hc_list[0]
    m_hc_u8 = torch.round(blended.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    del sub48_base, supp_base, hc_list, blended
    return m_hc_u8


def highpass_norm(img: np.ndarray, valid: np.ndarray, sigma: float = 42.0,
                  p_low: float = 52.0, p_high: float = 99.6, locked_bounds=None):
    f = img.astype(np.float32)
    bg = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    hp = np.maximum(0.0, f - bg * 0.88)
    vals = hp[valid]
    if vals.size < 100:
        return np.zeros_like(img, dtype=np.float32), (0.0, 1.0)
    if locked_bounds is not None:
        lo, hi = locked_bounds
    else:
        lo = float(np.percentile(vals, p_low))
        hi = float(np.percentile(vals, p_high))
    if hi <= lo + 1e-4:
        return np.zeros_like(img, dtype=np.float32), (lo, hi)
    out = np.clip((hp - lo) / (hi - lo), 0.0, 1.0) * 255.0
    out[~valid] = 0.0
    return out, (lo, hi)


def compute_recto_verso_and_periodicity(vol82: np.ndarray, valid_mask: np.ndarray, inward_sign: int, um: float) -> dict:
    """
    Computes:
    1. Bandpass Fiber Anisotropy A_RV(s) = E[(d_y I_bp)^2] / E[(d_x I_bp)^2] for s in {-3, 0, +3}
       (where d_y measures horizontal perfibrale fibers and d_x measures vertical transfibrale fibers).
    2. Dual-Face Differential delta_A_RV = A_RV(s_inward) - A_RV(s_outward) (H_H).
    3. Horizontal Line-Row Periodicity Score (H_F) in the 3.2 - 4.8 mm vertical pitch band.
    """
    def _a_rv_slice(sl_u8: np.ndarray) -> float:
        f = sl_u8.astype(np.float32)
        g1 = cv2.GaussianBlur(f, (0, 0), sigmaX=2.5, sigmaY=2.5)
        g2 = cv2.GaussianBlur(f, (0, 0), sigmaX=12.0, sigmaY=12.0)
        bp = g1 - g2
        gy = cv2.Sobel(bp, cv2.CV_32F, 0, 1, ksize=3)
        gx = cv2.Sobel(bp, cv2.CV_32F, 1, 0, ksize=3)
        ey = float(np.mean(gy[valid_mask] ** 2)) + 1e-5
        ex = float(np.mean(gx[valid_mask] ** 2)) + 1e-5
        return ey / ex

    s_mid = 41
    s_in = int(np.clip(s_mid + 3 * inward_sign, 0, 81))
    s_out = int(np.clip(s_mid - 3 * inward_sign, 0, 81))
    arv_mid = _a_rv_slice(vol82[s_mid])
    arv_in = _a_rv_slice(vol82[s_in])
    arv_out = _a_rv_slice(vol82[s_out])
    delta_arv = arv_in - arv_out

    # Line-row periodicity along Z (rows) on bandpass edge energy
    f_in = vol82[s_in].astype(np.float32)
    bp_in = cv2.GaussianBlur(f_in, (0, 0), 2.0) - cv2.GaussianBlur(f_in, (0, 0), 9.0)
    row_prof = np.std(bp_in, axis=1)
    row_prof = row_prof - gaussian_filter(row_prof, sigma=80.0)
    spec = np.abs(np.fft.rfft(row_prof))
    freqs = np.fft.rfftfreq(len(row_prof), d=um / 1000.0)  # cycles per mm
    # Target Greek line spacing: 3.2 to 4.8 mm -> 0.208 to 0.312 cycles/mm
    band = (freqs >= 0.205) & (freqs <= 0.315)
    bg_band = (freqs >= 0.08) & (freqs <= 0.60)
    periodicity_snr = float(spec[band].max() / (np.median(spec[bg_band]) + 1e-5)) if (np.any(band) and np.any(bg_band)) else 1.0

    return {
        "a_rv_mid": round(arv_mid, 4),
        "a_rv_inward_recto": round(arv_in, 4),
        "a_rv_outward_verso": round(arv_out, 4),
        "delta_a_rv_dual_face": round(delta_arv, 4),
        "is_recto_perfibrale": bool(max(arv_in, arv_mid) >= 1.06 or delta_arv >= 0.02),
        "row_periodicity_snr_3_5mm": round(periodicity_snr, 3),
    }


def evaluate_stroke_mask_attenuation(
    dn_star: np.ndarray,
    hc_star: np.ndarray,
    dn_m4: np.ndarray,
    hc_m4: np.ndarray,
    dn_p4: np.ndarray,
    hc_p4: np.ndarray,
    valid_mask: np.ndarray,
) -> dict:
    """
    Computes Kernel-Matched Thin-Sheet Surface Derivative (`I_thin`, Bet 6) and evaluates
    the Frozen-Stroke-Mask Attenuation on `M_stroke` between `d*` and `d* +- 4` voxels!
    """
    dn_f0 = dn_star.astype(np.float32)
    hc_f0 = hc_star.astype(np.float32)
    dn_fm4 = dn_m4.astype(np.float32)
    hc_fm4 = hc_m4.astype(np.float32)
    dn_fp4 = dn_p4.astype(np.float32)
    hc_fp4 = hc_p4.astype(np.float32)

    # 1. Raw model Z-attenuation at top 2% bright consensus pixels on d*
    dn_hp0, dn_bounds = highpass_norm(dn_f0, valid_mask, sigma=42.0, p_low=52.0, p_high=99.6)
    hc_hp0, hc_bounds = highpass_norm(hc_f0, valid_mask, sigma=50.0, p_low=50.0, p_high=99.6)
    # Lock scale bounds to d* so off-lamina slices are NEVER artificially stretched!
    dn_hpm4, _ = highpass_norm(dn_fm4, valid_mask, sigma=42.0, locked_bounds=dn_bounds)
    dn_hpp4, _ = highpass_norm(dn_fp4, valid_mask, sigma=42.0, locked_bounds=dn_bounds)
    hc_hpm4, _ = highpass_norm(hc_fm4, valid_mask, sigma=50.0, locked_bounds=hc_bounds)
    hc_hpp4, _ = highpass_norm(hc_fp4, valid_mask, sigma=50.0, locked_bounds=hc_bounds)

    # 2. Kernel-Matched Z-DoG (alpha_villa = 0.75, alpha_hecate = 0.90)
    dn_thin = np.maximum(0.0, dn_hp0 - 0.75 * 0.5 * (dn_hpm4 + dn_hpp4))
    hc_thin = np.maximum(0.0, hc_hp0 - 0.90 * 0.5 * (hc_hpm4 + hc_hpp4))

    # Scale-locked multiplicative thinness gate W_thin(y, x)
    ens_hp0 = 0.50 * dn_hp0 + 0.50 * hc_hp0
    ens_hpm4 = 0.50 * dn_hpm4 + 0.50 * hc_hpm4
    ens_hpp4 = 0.50 * dn_hpp4 + 0.50 * hc_hpp4
    w_thin = np.clip(
        (ens_hp0 - 0.72 * 0.5 * (ens_hpm4 + ens_hpp4)) / np.maximum(0.32 * ens_hp0, 4.0),
        0.0,
        1.0,
    )
    dn_thin_gated = (dn_hp0 * w_thin).astype(np.uint8)
    hc_thin_gated = (hc_hp0 * w_thin).astype(np.uint8)
    ens_thin_gated = (ens_hp0 * w_thin).astype(np.uint8)

    # Correlation of thin-gated maps
    v_dn_t = dn_thin_gated[valid_mask].astype(np.float32)
    v_hc_t = hc_thin_gated[valid_mask].astype(np.float32)
    r_thin = float(np.corrcoef(v_dn_t, v_hc_t)[0, 1]) if (v_dn_t.size > 500 and v_dn_t.std() > 1e-3 and v_hc_t.std() > 1e-3) else 0.0

    # Frozen-Stroke-Mask M_stroke: pixels where thin calligraphic consensus is in top 1.5% AND both models active
    if v_dn_t.size > 500:
        thr_t = max(float(np.percentile(ens_thin_gated[valid_mask], 98.5)), 25.0)
        m_stroke = valid_mask & (ens_thin_gated >= thr_t) & (dn_hp0 >= 35.0) & (hc_hp0 >= 35.0)
        if m_stroke.sum() < 80:
            thr_raw = max(float(np.percentile(ens_hp0[valid_mask], 98.5)), 35.0)
            m_stroke = valid_mask & (ens_hp0 >= thr_raw)
    else:
        m_stroke = valid_mask

    if m_stroke.sum() >= 20:
        i0 = float(ens_hp0[m_stroke].mean())
        im4 = float(ens_hpm4[m_stroke].mean())
        ip4 = float(ens_hpp4[m_stroke].mean())
        drop_m4 = (i0 - im4) / max(i0, 1.0) * 100.0
        drop_p4 = (i0 - ip4) / max(i0, 1.0) * 100.0
        drop_pm4 = (i0 - 0.5 * (im4 + ip4)) / max(i0, 1.0) * 100.0
        hc_i0 = float(hc_hp0[m_stroke].mean())
        hc_im4 = float(hc_hpm4[m_stroke].mean())
        hc_ip4 = float(hc_hpp4[m_stroke].mean())
        hc_drop_pm4 = (hc_i0 - 0.5 * (hc_im4 + hc_ip4)) / max(hc_i0, 1.0) * 100.0
    else:
        i0, im4, ip4, drop_m4, drop_p4, drop_pm4, hc_drop_pm4 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return {
        "r_thin_2x2cm": round(r_thin, 4),
        "stroke_mask_px": int(m_stroke.sum()),
        "stroke_i0": round(i0, 2),
        "stroke_im4": round(im4, 2),
        "stroke_ip4": round(ip4, 2),
        "stroke_drop_m4_pct": round(drop_m4, 1),
        "stroke_drop_p4_pct": round(drop_p4, 1),
        "stroke_atten_pm4_pct": round(drop_pm4, 1),
        "hecate_stroke_atten_pm4_pct": round(hc_drop_pm4, 1),
        "dn_thin_u8": dn_thin_gated,
        "hc_thin_u8": hc_thin_gated,
        "ens_thin_u8": ens_thin_gated,
    }


job = json.load(open("/content/current_job.json"))
scroll_key = job["scroll_key"]
cand_id = job["cand_id"]
pool = job["pool"]
ct_zarr_url = job["ct_zarr_url"]
volZ, volY, volX = job["shape_zyx"]
um = float(job["um"])
inward_sign = int(job.get("inward_normal_sign", 1))

chunks_dir = f"/content/chunks/{scroll_key}"
os.makedirs(chunks_dir, exist_ok=True)
m_dir = Path(f"/content/meshes/stage_i_m7snap/{scroll_key}/{cand_id}")
t_seg = time.time()
print(f"\n=== [STAGE I SCOUT] {pool} | {scroll_key}/{cand_id} (inward_sign={inward_sign:+d}) ===", flush=True)

meta = json.load(open(m_dir / "meta.json"))
snap_stats = meta.get("stage_i_m7_snap", {})
x = tifffile.imread(str(m_dir / "x.tif")).astype(np.float32)
y = tifffile.imread(str(m_dir / "y.tif")).astype(np.float32)
z = tifffile.imread(str(m_dir / "z.tif")).astype(np.float32)
rows, cols = x.shape
x.tofile(str(m_dir / "x.bin"))
y.tofile(str(m_dir / "y.bin"))
z.tofile(str(m_dir / "z.bin"))
chunks_txt = str(m_dir / "chunks_82.txt")

subprocess.run([
    "/content/code/render_tifxyz_82", "--list-chunks",
    str(m_dir / "x.bin"), str(m_dir / "y.bin"), str(m_dir / "z.bin"),
    str(rows), str(cols), str(volZ), str(volY), str(volX), chunks_txt
], check=True)
subprocess.run(["python3", "/content/code/fetch_s3_chunks.py", ct_zarr_url, chunks_txt, chunks_dir, "64"], check=True)

zarr_dir = Path(f"/content/work/{scroll_key}/{cand_id}_82.zarr")
subprocess.run([
    "/content/code/render_tifxyz_82", "--render",
    str(m_dir / "x.bin"), str(m_dir / "y.bin"), str(m_dir / "z.bin"),
    str(rows), str(cols), str(volZ), str(volY), str(volX), chunks_dir, str(zarr_dir)
], check=True)

H, W = rows * 20, cols * 20
vol82 = np.zeros((82, H, W), dtype=np.uint8)
tilesY, tilesX = (H + 127) // 128, (W + 127) // 128
for ty in range(tilesY):
    y0, dy = ty * 128, min(128, H - ty * 128)
    for tx in range(tilesX):
        x0, dx = tx * 128, min(128, W - tx * 128)
        cp = zarr_dir / "0" / "0" / str(ty) / str(tx)
        if cp.exists():
            buf = np.fromfile(str(cp), dtype=np.uint8).reshape(82, 128, 128)
            vol82[:, y0:y0 + dy, x0:x0 + dx] = buf[:, :dy, :dx]
shutil.rmtree(zarr_dir, ignore_errors=True)

valid_mask = (vol82[41] > 0)
rv_stats = compute_recto_verso_and_periodicity(vol82, valid_mask, inward_sign, um)
print(
    f"  [CT Stratigraphy] A_RV(recto)={rv_stats['a_rv_inward_recto']:.3f} vs A_RV(verso)={rv_stats['a_rv_outward_verso']:.3f} "
    f"(delta={rv_stats['delta_a_rv_dual_face']:+.3f}) | row_periodicity_SNR={rv_stats['row_periodicity_snr_3_5mm']:.2f}",
    flush=True,
)

out_dir = Path(f"/content/results/{scroll_key}/{cand_id}")
out_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_dir / "ct_mid.png"), vol82[41])

# Stage 4A: Test reverse vs forward at d = 0 (fast duet dnative + Hecate at 1x TTA)
vol_d0 = vol82[8:8 + 66]
best_orient = "reverse"
best_r0 = -1.0
d0_cache = {}
for cand_orient, rev_flag in [("reverse", True), ("forward", False)]:
    z0_v = 26 if rev_flag else 23
    sub17 = vol_d0[z0_v:z0_v + 17]
    sub17 = sub17[::-1].copy() if rev_flag else sub17.copy()
    supp_2d = sub17.any(axis=0)
    m_dn = eval_dnative_only_cuda(sub17, supp_2d, n_rot=1)
    m_hc = eval_hecate_tta_cuda(vol_d0, supp_2d, reverse=rev_flag, n_rot=1)
    v_dn = m_dn[valid_mask].astype(np.float32)
    v_hc = m_hc[valid_mask].astype(np.float32)
    r_raw = float(np.corrcoef(v_dn, v_hc)[0, 1]) if (v_dn.size > 500 and v_dn.std() > 1e-3 and v_hc.std() > 1e-3) else 0.0
    dn_hp, _ = highpass_norm(m_dn, valid_mask, sigma=42.0)
    hc_hp, _ = highpass_norm(m_hc, valid_mask, sigma=50.0)
    r_hp = float(np.corrcoef(dn_hp[valid_mask], hc_hp[valid_mask])[0, 1]) if v_dn.size > 500 else r_raw
    sc = 0.55 * r_raw + 0.45 * r_hp
    d0_cache[cand_orient] = (m_dn, m_hc, r_raw, r_hp)
    if sc > best_r0:
        best_r0 = sc
        best_orient = cand_orient

rev_flag = (best_orient == "reverse")
print(
    f"  [Orient Lock d=0] Selected {best_orient}: r_raw={d0_cache[best_orient][2]:.4f}, r_hp={d0_cache[best_orient][3]:.4f}",
    flush=True,
)

# Stage 4B: Evaluate 7-point 2-voxel Scout Stack d in {-6, -4, -2, 0, +2, +4, +6}
# So any candidate surface d* in {-2, 0, +2} has exact symmetric d* - 4 and d* + 4 layers!
scout_depths = [-6, -4, -2, 0, +2, +4, +6]
dn_map_by_d = {}
hc_map_by_d = {}
ct_map_by_d = {}
depth_results = {}

for d in scout_depths:
    s0 = 8 + d
    vol_d = vol82[s0:s0 + 66]
    ct_map_by_d[d] = vol_d[34].copy()
    if d == 0:
        m_dn, m_hc, r_raw, r_hp = d0_cache[best_orient]
    else:
        z0_v = 26 if rev_flag else 23
        sub17 = vol_d[z0_v:z0_v + 17]
        sub17 = sub17[::-1].copy() if rev_flag else sub17.copy()
        supp_2d = sub17.any(axis=0)
        m_dn = eval_dnative_only_cuda(sub17, supp_2d, n_rot=1)
        m_hc = eval_hecate_tta_cuda(vol_d, supp_2d, reverse=rev_flag, n_rot=1)
        v_dn = m_dn[valid_mask].astype(np.float32)
        v_hc = m_hc[valid_mask].astype(np.float32)
        r_raw = float(np.corrcoef(v_dn, v_hc)[0, 1]) if (v_dn.size > 500 and v_dn.std() > 1e-3 and v_hc.std() > 1e-3) else 0.0
        dn_hp, _ = highpass_norm(m_dn, valid_mask, sigma=42.0)
        hc_hp, _ = highpass_norm(m_hc, valid_mask, sigma=50.0)
        r_hp = float(np.corrcoef(dn_hp[valid_mask], hc_hp[valid_mask])[0, 1]) if v_dn.size > 500 else r_raw

    dn_map_by_d[d] = m_dn
    hc_map_by_d[d] = m_hc
    v_dn = m_dn[valid_mask].astype(np.float32)
    v_hc = m_hc[valid_mask].astype(np.float32)
    ens2 = 0.5 * (v_dn + v_hc)
    p50, p995 = [float(x) for x in np.percentile(ens2, [50.0, 99.5])] if v_dn.size > 500 else (0.0, 0.0)
    hc_p995 = float(np.percentile(v_hc, 99.5)) if v_hc.size > 500 else 0.0
    depth_results[f"d{d:+d}"] = {
        "r_2x2cm": round(r_raw, 4),
        "r_hp_2x2cm": round(r_hp, 4),
        "ens_p50": round(p50, 2),
        "ens_p995": round(p995, 2),
        "hecate_p995": round(hc_p995, 2),
    }

# Evaluate Thin-Sheet Surface Derivative (`I_thin`) at each candidate surface depth d* in {-2, 0, +2}
# (with exact symmetric d*-4 and d*+4 from scout_depths!)
best_dstar = 0
best_thin_eval = None
best_comb_score = -1e9

for d_cand in [-2, 0, +2]:
    teval = evaluate_stroke_mask_attenuation(
        dn_map_by_d[d_cand], hc_map_by_d[d_cand],
        dn_map_by_d[d_cand - 4], hc_map_by_d[d_cand - 4],
        dn_map_by_d[d_cand + 4], hc_map_by_d[d_cand + 4],
        valid_mask,
    )
    r_c = max(depth_results[f"d{d_cand:+d}"]["r_2x2cm"], depth_results[f"d{d_cand:+d}"]["r_hp_2x2cm"], teval["r_thin_2x2cm"])
    # Bonus if d_cand aligns with the inward recto face of the sheet!
    recto_bonus = 0.03 if (d_cand * inward_sign >= 0) else 0.0
    comb = r_c + 0.008 * max(0.0, teval["stroke_atten_pm4_pct"]) + recto_bonus
    depth_results[f"d{d_cand:+d}"]["stroke_atten_pm4_pct"] = teval["stroke_atten_pm4_pct"]
    depth_results[f"d{d_cand:+d}"]["r_thin_2x2cm"] = teval["r_thin_2x2cm"]
    if comb > best_comb_score:
        best_comb_score = comb
        best_dstar = d_cand
        best_thin_eval = teval

best_d_key = f"d{best_dstar:+d}"
r_best_raw = depth_results[best_d_key]["r_2x2cm"]
r_best_hp = depth_results[best_d_key]["r_hp_2x2cm"]
r_best_thin = best_thin_eval["r_thin_2x2cm"]
atten_pm4 = best_thin_eval["stroke_atten_pm4_pct"]
hc_atten_pm4 = best_thin_eval["hecate_stroke_atten_pm4_pct"]

print(
    f"  [Scout Result d*={best_d_key}] r_raw={r_best_raw:.4f} r_hp={r_best_hp:.4f} r_thin={r_best_thin:.4f} "
    f"| stroke_atten_pm4={atten_pm4:.1f}% (Hecate={hc_atten_pm4:.1f}%)",
    flush=True,
)

# Check Auto-Promotion Gate (Bet 7 & Bet 8):
# Promote if r >= 0.24 AND (stroke_atten_pm4 >= 30.0% OR hecate_stroke_atten_pm4 >= 45.0%)
promoted = bool(
    max(r_best_raw, r_best_hp, r_best_thin) >= 0.24
    and (atten_pm4 >= 30.0 or hc_atten_pm4 >= 45.0)
)

if promoted:
    print(f"  *** AUTO-PROMOTING {scroll_key}/{cand_id} (d*={best_d_key}) TO 4x TTA + SEED42/43 DENSE STACK ***", flush=True)
    promo_depths = sorted(set(scout_depths + [best_dstar - 1, best_dstar + 1]))
    promo_depths = [d for d in promo_depths if -6 <= d <= 6]
    for d in promo_depths:
        s0 = 8 + d
        vol_d = vol82[s0:s0 + 66]
        ct_map_by_d[d] = vol_d[34].copy()
        z0_v = 26 if rev_flag else 23
        sub17 = vol_d[z0_v:z0_v + 17]
        sub17 = sub17[::-1].copy() if rev_flag else sub17.copy()
        supp_2d = sub17.any(axis=0)
        v_nrot = 4 if abs(d - best_dstar) <= 1 else 2
        h_nrot = 2 if abs(d - best_dstar) <= 1 else 1
        m_dn = eval_dnative_only_cuda(sub17, supp_2d, n_rot=v_nrot)
        m_42, m_43 = eval_seed42_43_cuda(sub17, supp_2d, n_rot=max(1, v_nrot // 2))
        m_hc = eval_hecate_tta_cuda(vol_d, supp_2d, reverse=rev_flag, n_rot=h_nrot)
        # Blend Villa 3-model average into dn_map_by_d for full 4-model parity
        villa_u8 = np.clip(0.45 * m_dn.astype(np.float32) + 0.275 * m_42.astype(np.float32) + 0.275 * m_43.astype(np.float32), 0.0, 255.0).astype(np.uint8)
        dn_map_by_d[d] = villa_u8
        hc_map_by_d[d] = m_hc
        v_v = villa_u8[valid_mask].astype(np.float32)
        v_h = m_hc[valid_mask].astype(np.float32)
        r_raw = float(np.corrcoef(v_v, v_h)[0, 1]) if (v_v.size > 500 and v_v.std() > 1e-3 and v_h.std() > 1e-3) else 0.0
        dn_hp, _ = highpass_norm(villa_u8, valid_mask, sigma=42.0)
        hc_hp, _ = highpass_norm(m_hc, valid_mask, sigma=50.0)
        r_hp = float(np.corrcoef(dn_hp[valid_mask], hc_hp[valid_mask])[0, 1]) if v_v.size > 500 else r_raw
        ens_f = 0.5 * (v_v + v_h)
        p50, p995 = [float(x) for x in np.percentile(ens_f, [50.0, 99.5])] if v_v.size > 500 else (0.0, 0.0)
        depth_results[f"d{d:+d}"] = {
            "r_2x2cm": round(r_raw, 4),
            "r_hp_2x2cm": round(r_hp, 4),
            "ens_p50": round(p50, 2),
            "ens_p995": round(p995, 2),
            "hecate_p995": round(float(np.percentile(v_h, 99.5)), 2),
        }
    # Re-evaluate thin-sheet derivative at best_dstar after 4x TTA + seed42/43 promotion
    best_thin_eval = evaluate_stroke_mask_attenuation(
        dn_map_by_d[best_dstar], hc_map_by_d[best_dstar],
        dn_map_by_d[best_dstar - 4], hc_map_by_d[best_dstar - 4],
        dn_map_by_d[best_dstar + 4], hc_map_by_d[best_dstar + 4],
        valid_mask,
    )
    r_best_raw = depth_results[best_d_key]["r_2x2cm"]
    r_best_hp = depth_results[best_d_key]["r_hp_2x2cm"]
    r_best_thin = best_thin_eval["r_thin_2x2cm"]
    atten_pm4 = best_thin_eval["stroke_atten_pm4_pct"]
    hc_atten_pm4 = best_thin_eval["hecate_stroke_atten_pm4_pct"]

# Scale-Locked 3D Stack Z-Score Normalization across all evaluated depths (NO per-layer stretching!)
all_d_sorted = sorted(dn_map_by_d.keys())
dn_arr = np.stack([dn_map_by_d[d].astype(np.float32) for d in all_d_sorted], axis=0)
hc_arr = np.stack([hc_map_by_d[d].astype(np.float32) for d in all_d_sorted], axis=0)

def global_znorm_3d(arr3d: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    vals = arr3d[:, mask2d]
    if vals.size < 100:
        return np.zeros_like(arr3d)
    mu = float(vals.mean())
    sd = max(float(vals.std()), 1e-4)
    return (arr3d - mu) / sd

ens_z_3d = 0.50 * global_znorm_3d(dn_arr, valid_mask) + 0.50 * global_znorm_3d(hc_arr, valid_mask)
z_vals_all = ens_z_3d[:, valid_mask]
lo_z_3d, hi_z_3d = np.percentile(z_vals_all, [1.0, 99.9]) if z_vals_all.size > 100 else (-0.7, 3.8)
hi_z_3d = max(float(hi_z_3d), float(lo_z_3d) + 1.0)

for idx_d, d in enumerate(all_d_sorted):
    d_key = f"d{d:+d}"
    z_sl = ens_z_3d[idx_d]
    ens_u8 = (np.clip((z_sl - lo_z_3d) / (hi_z_3d - lo_z_3d), 0.0, 1.0) * 255.0).astype(np.uint8)
    ens_u8[~valid_mask] = 0
    cv2.imwrite(str(out_dir / f"ens_z_{best_orient}_{d_key}.png"), ens_u8)
    cv2.imwrite(str(out_dir / f"dnative_{best_orient}_{d_key}.png"), dn_map_by_d[d])
    cv2.imwrite(str(out_dir / f"hecate_{best_orient}_{d_key}.png"), hc_map_by_d[d])
    depth_results[d_key]["global_z_p995"] = round(float(np.percentile(z_sl[valid_mask], 99.5)), 4)

# Save thin-layer gated maps at best_dstar for the Stage I paleography detector!
cv2.imwrite(str(out_dir / "ct_mid_single_dstar.png"), ct_map_by_d[best_dstar])
cv2.imwrite(str(out_dir / f"dnative_{best_orient}_thin_dstar.png"), best_thin_eval["dn_thin_u8"])
cv2.imwrite(str(out_dir / f"hecate_{best_orient}_thin_dstar.png"), best_thin_eval["hc_thin_u8"])
cv2.imwrite(str(out_dir / f"ens_z_{best_orient}_thin_dstar.png"), best_thin_eval["ens_thin_u8"])

# Render Gold overlay of the thin-layer gated surface ink
alpha_t = 0.9 * np.clip((best_thin_eval["ens_thin_u8"].astype(np.float32) / 255.0 - 0.25) / 0.45, 0.0, 1.0)
alpha_t[~valid_mask] = 0.0
ct_f = ct_map_by_d[best_dstar].astype(np.float32)[..., None]
gold = np.array([255.0, 235.0, 120.0], dtype=np.float32)[None, None, :]
gold_rgb = np.clip(ct_f * 0.45 * (1.0 - alpha_t[..., None]) + gold * alpha_t[..., None], 0.0, 255.0).astype(np.uint8)
cv2.imwrite(str(out_dir / f"submission_gold_{best_orient}_thin_dstar.png"), cv2.cvtColor(gold_rgb, cv2.COLOR_RGB2BGR))

seg_report = {
    "scroll_key": scroll_key,
    "cand_id": cand_id,
    "pool": pool,
    "um": um,
    "shape_hw": [int(H), int(W)],
    "chosen_orient": best_orient,
    "inward_normal_sign": inward_sign,
    "promoted_to_4x_tta": promoted,
    "stage_i_m7_snap": snap_stats,
    "ct_stratigraphy": rv_stats,
    "best_single_depth": best_d_key,
    "r_raw_2x2cm_dstar": round(r_best_raw, 4),
    "r_hp_2x2cm_dstar": round(r_best_hp, 4),
    "r_thin_2x2cm_dstar": round(r_best_thin, 4),
    "stroke_atten_pm4_pct": round(atten_pm4, 1),
    "hecate_stroke_atten_pm4_pct": round(hc_atten_pm4, 1),
    "stroke_drop_m4_pct": best_thin_eval["stroke_drop_m4_pct"],
    "stroke_drop_p4_pct": best_thin_eval["stroke_drop_p4_pct"],
    "depths": depth_results,
    "elapsed_s": round(time.time() - t_seg, 1),
}
with open(out_dir / "metrics_stage_i.json", "w") as f:
    json.dump(seg_report, f, indent=2)

print(
    f"  ==> [DONE {scroll_key}/{cand_id}] d*={best_d_key} | r_raw={r_best_raw:.4f} r_hp={r_best_hp:.4f} r_thin={r_best_thin:.4f} "
    f"| stroke_atten_pm4={atten_pm4:.1f}% (Hecate={hc_atten_pm4:.1f}%) | promoted={promoted} ({seg_report['elapsed_s']}s)",
    flush=True,
)
subprocess.run(["tar", "-czf", "/content/seg_out.tar.gz", "-C", "/content/results", f"{scroll_key}/{cand_id}"], check=True)
print("SEGMENT_COMPLETE_OK", flush=True)
'''


def run_colab(config_path: Path, args: list, check: bool = True):
    cmd = [COLAB_BIN, "--config", str(config_path)] + args
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu", type=str, default="A100")
    args = parser.parse_args()

    manifest = json.loads((BASE_DIR / "meshes" / "stage_i_scout_manifest.json").read_text())
    from stage_i_prepare_36_candidates import SCROLL_CONFIGS

    # Filter candidates that pass the CPU m7-Snap gate (or sort by m7_prob_mean_after * m7_ridge_attenuation_pm4_pct)
    cands = manifest["candidates"]
    cands_sorted = sorted(
        cands,
        key=lambda m: (
            1 if m["pool"] == "Pool_A_77keV_L2" else 0,
            m["stage_i_m7_snap"]["m7_prob_mean_after"] * (m["stage_i_m7_snap"]["m7_ridge_attenuation_pm4_pct"] / 100.0),
        ),
        reverse=True,
    )[:38]

    # Shard across workers
    my_cands = [c for i, c in enumerate(cands_sorted) if (i % args.num_workers) == args.worker_id]
    w = args.worker_id
    session_name = f"vesuvius-i{w}"
    cfg_dir = Path(f"/tmp/colab_i{w}")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "sessions.json"
    if cfg_path.exists():
        cfg_path.unlink()

    out_root = BASE_DIR / "results_stage_i" / f"w{w}"
    out_root.mkdir(parents=True, exist_ok=True)

    tar_path = cfg_dir / f"worker_input_i{w}.tar.gz"
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
    seg_runner_py.write_text(REMOTE_SCOUT_RUNNER)

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
            done_json = out_root / sc_key / cid / "metrics_stage_i.json"
            if done_json.exists():
                print(f"[skip] {sc_key}/{cid} already downloaded locally at {done_json}", flush=True)
                continue

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

            print(f"\n[Worker {w}] ({idx_c+1}/{len(my_cands)}) Scouting {sc_key}/{cid} ...", flush=True)
            for attempt in range(2):
                res = run_colab(cfg_path, ["exec", "--timeout", "1800", "-s", session_name, "-f", str(seg_runner_py)], check=False)
                if res.returncode == 0:
                    break
                print(f"[warn] attempt {attempt+1} on {sc_key}/{cid} returned {res.returncode}, retrying...", flush=True)
                time.sleep(3)

            local_seg_tar = cfg_dir / f"seg_{sc_key}_{cid}.tar.gz"
            if local_seg_tar.exists():
                local_seg_tar.unlink()
            dl_res = run_colab(cfg_path, ["download", "-s", session_name, "/content/seg_out.tar.gz", str(local_seg_tar)], check=False)
            if dl_res.returncode == 0 and local_seg_tar.exists():
                with tarfile.open(local_seg_tar, "r:gz") as tf:
                    tf.extractall(path=out_root)
                print(f"[saved] Unpacked Stage I {sc_key}/{cid} -> {out_root / sc_key / cid}", flush=True)
            else:
                print(f"[error] Failed to download /content/seg_out.tar.gz for {sc_key}/{cid}", flush=True)
    finally:
        run_colab(cfg_path, ["stop", "-s", session_name], check=False)


if __name__ == "__main__":
    main()
