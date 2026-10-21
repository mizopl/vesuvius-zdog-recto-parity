#!/usr/bin/env python3
"""Scan a rendered 66-layer surface volume zarr on Apple Silicon MPS across both normal
orientations (forward & reverse) and multiple depth windows, using our 3 Villa models
(dnative, hybrid_3d2d-seed42, hybrid_3d2d-seed43) + optional Hecate 3D.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import zarr

BASE_343 = Path('/workspace/v3r1t4s/runs/2026-10-19-0910')
sys.path.insert(0, str(BASE_343 / 'data/villa/vesuvius/src'))
sys.path.insert(0, str(BASE_343 / 'work/ckpt'))

from vesuvius.ink_detection.inference.infer import (
    _sliding_positions_1d,
    compute_importance_map_2d,
    configure_model,
    logits_to_probabilities,
    normalize_flat_patch,
)

CKPT_DIR = BASE_343 / 'work/ckpt'


def create_zarr_metadata(zarr_dir: Path, num_slices: int, H: int, W: int):
    zarr_dir.mkdir(parents=True, exist_ok=True)
    (zarr_dir / '.zgroup').write_text('{"zarr_format": 2}\n')
    z0 = zarr_dir / '0'
    z0.mkdir(parents=True, exist_ok=True)
    zarray = {
        "chunks": [num_slices, 128, 128],
        "compressor": None,
        "dimension_separator": "/",
        "dtype": "|u1",
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": [num_slices, H, W],
        "zarr_format": 2,
    }
    (z0 / '.zarray').write_text(json.dumps(zarray, indent=2) + '\n')


def eval_villa_single(model, preproc, weight_map, p: int, vol17: np.ndarray, bs: int = 16) -> np.ndarray:
    Vh, Vw = vol17.shape[1], vol17.shape[2]
    stride = p // 2
    ys = _sliding_positions_1d(Vh, p, stride)
    xs = _sliding_positions_1d(Vw, p, stride)
    coords = []
    for y0 in ys:
        vh = min(p, Vh - y0)
        for x0 in xs:
            vw = min(p, Vw - x0)
            sub = vol17[:, y0:y0 + vh, x0:x0 + vw]
            if not sub.any():
                continue
            coords.append((y0, x0, vh, vw))

    prob_sum = np.zeros((Vh, Vw), dtype=np.float32)
    weight_sum = np.zeros((Vh, Vw), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(coords), bs):
            batch_coords = coords[start:start + bs]
            batch_tensors = []
            for (y0, x0, vh, vw) in batch_coords:
                patch = np.zeros((17, p, p), dtype=vol17.dtype)
                patch[:, :vh, :vw] = vol17[:, y0:y0 + vh, x0:x0 + vw]
                normed = normalize_flat_patch(patch, preproc)
                batch_tensors.append(normed[None])
            x_batch = torch.from_numpy(np.stack(batch_tensors)).to('mps')
            probs = logits_to_probabilities(model(x_batch), image_hw=(p, p)).cpu().numpy()[:, 0]
            for prob, (y0, x0, vh, vw) in zip(probs, batch_coords):
                w = weight_map[:vh, :vw]
                prob_sum[y0:y0 + vh, x0:x0 + vw] += prob[:vh, :vw] * w
                weight_sum[y0:y0 + vh, x0:x0 + vw] += w
    out = np.zeros((Vh, Vw), dtype=np.float32)
    np.divide(prob_sum, weight_sum, out=out, where=weight_sum > 1e-6)
    np.clip(out, 0.0, 1.0, out=out)
    return (out * 255.0).astype(np.uint8)


def make_overlay(ct_slice: np.ndarray, ink_u8: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    ct_eq = clahe.apply(ct_slice)
    rgb = np.stack([ct_eq, ct_eq, ct_eq], axis=-1).astype(np.float32)
    alpha = np.clip((ink_u8.astype(np.float32) - 25.0) / 180.0, 0.0, 1.0)[..., None]
    red = np.zeros_like(rgb)
    red[..., 0] = 255.0  # R in RGB
    red[..., 1] = 40.0
    red[..., 2] = 20.0
    comp = (rgb * (1.0 - 0.78 * alpha) + red * (0.78 * alpha)).clip(0, 255).astype(np.uint8)
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr_dir", type=Path)
    ap.add_argument("rows", type=int)
    ap.add_argument("cols", type=int)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    H = args.rows * 20
    W = args.cols * 20
    create_zarr_metadata(args.zarr_dir, 66, H, W)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening {args.zarr_dir} (shape=66x{H}x{W}) ...", flush=True)
    st = zarr.open(str(args.zarr_dir / '0'), mode='r')
    vol66 = np.asarray(st[:])
    valid_mask = vol66[32] > 0

    # Save mid-plane CT projection (mean of slices 29..37)
    ct_mid = np.mean(vol66[29:37], axis=0).astype(np.uint8)
    cv2.imwrite(str(args.out_dir / "ct_mid.png"), ct_mid)

    models_spec = [
        ('dnative', CKPT_DIR / 'dense_native-016000.pth'),
        ('seed42', CKPT_DIR / 'hybrid_3d2d-seed42_step-075000.pth'),
        ('seed43', CKPT_DIR / 'hybrid_3d2d-seed43_step-075000.pth'),
    ]

    # Step 1: Load dnative and sweep depths in BOTH orientations (forward and reverse)
    t0 = time.time()
    cfg = configure_model(argparse.Namespace(checkpoint=models_spec[0][1], amp_dtype='default'))
    model = cfg.model.to('mps').eval()
    p = int(cfg.patch_size)
    preproc = cfg.preprocessing
    weight_map = compute_importance_map_2d(patch_size=(p, p), mode='hann').numpy()

    best_orient = None
    best_z0 = None
    best_score = -1.0
    sweep_stats = []

    for orient in ['reverse', 'forward']:
        for z0 in [20, 23, 25, 27, 30]:
            sub = vol66[z0:z0 + 17]
            if orient == 'reverse':
                sub = sub[::-1].copy()
            else:
                sub = sub.copy()
            pred = eval_villa_single(model, preproc, weight_map, p, sub)
            # Score: 99.5th percentile + fraction of pixels > 80 inside valid mask
            vpred = pred[valid_mask]
            if vpred.size == 0:
                continue
            p995 = float(np.percentile(vpred, 99.5))
            frac80 = float(np.mean(vpred > 80) * 100.0)
            frac120 = float(np.mean(vpred > 120) * 100.0)
            score = p995 + 20.0 * frac80
            sweep_stats.append({
                "orient": orient,
                "z0": z0,
                "p995": round(p995, 2),
                "frac80_pct": round(frac80, 3),
                "frac120_pct": round(frac120, 3),
                "score": round(score, 2),
            })
            print(f"  [dnative sweep] {orient:7s} z={z0:2d}..{z0+17:2d} | p99.5={p995:5.1f} | >80={frac80:5.2f}% | >120={frac120:5.2f}%", flush=True)
            cv2.imwrite(str(args.out_dir / f"dnative_{orient}_z{z0}.png"), pred)
            if score > best_score:
                best_score = score
                best_orient = orient
                best_z0 = z0

    del model
    torch.mps.empty_cache()

    # Step 2: Run seed42 and seed43 on BOTH standard windows (reverse z=26, forward z=23) AND best_orient/best_z0
    windows_to_run = {
        ('reverse', 26),
        ('forward', 23),
    }
    if best_orient is not None:
        windows_to_run.add((best_orient, best_z0))

    preds = {}
    for orient, z0 in sorted(windows_to_run):
        dnative_path = args.out_dir / f"dnative_{orient}_z{z0}.png"
        if dnative_path.exists():
            preds[('dnative', orient, z0)] = cv2.imread(str(dnative_path), cv2.IMREAD_GRAYSCALE)

    for mname, ckpt_path in models_spec:
        need = [(o, z) for (o, z) in sorted(windows_to_run) if (mname, o, z) not in preds]
        if not need:
            continue
        cfg = configure_model(argparse.Namespace(checkpoint=ckpt_path, amp_dtype='default'))
        model = cfg.model.to('mps').eval()
        p = int(cfg.patch_size)
        preproc = cfg.preprocessing
        weight_map = compute_importance_map_2d(patch_size=(p, p), mode='hann').numpy()
        for orient, z0 in need:
            sub = vol66[z0:z0 + 17]
            sub = sub[::-1].copy() if orient == 'reverse' else sub.copy()
            t_m = time.time()
            pred = eval_villa_single(model, preproc, weight_map, p, sub)
            preds[(mname, orient, z0)] = pred
            cv2.imwrite(str(args.out_dir / f"{mname}_{orient}_z{z0}.png"), pred)
            print(f"  [{mname}] {orient} z={z0} done in {time.time() - t_m:.1f}s", flush=True)
        del model
        torch.mps.empty_cache()

    # Build 3-model ensembles and overlays for each evaluated window
    summary = {"shape_yx": [H, W], "valid_pixels": int(valid_mask.sum()), "sweep": sweep_stats, "ensembles": []}
    for orient, z0 in sorted(windows_to_run):
        m_dn = preds[('dnative', orient, z0)].astype(np.float32)
        m_42 = preds[('seed42', orient, z0)].astype(np.float32)
        m_43 = preds[('seed43', orient, z0)].astype(np.float32)
        ens = np.clip(0.40 * m_dn + 0.30 * m_42 + 0.30 * m_43, 0, 255).astype(np.uint8)
        ens_path = args.out_dir / f"ensemble_{orient}_z{z0}.png"
        cv2.imwrite(str(ens_path), ens)
        ov = make_overlay(ct_mid, ens)
        ov_path = args.out_dir / f"overlay_{orient}_z{z0}.png"
        cv2.imwrite(str(ov_path), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        vens = ens[valid_mask]
        p995 = float(np.percentile(vens, 99.5)) if vens.size else 0.0
        frac80 = float(np.mean(vens > 80) * 100.0) if vens.size else 0.0
        frac100 = float(np.mean(vens > 100) * 100.0) if vens.size else 0.0
        summary["ensembles"].append({
            "orient": orient,
            "z0": z0,
            "ens_p995": round(p995, 2),
            "ens_frac80_pct": round(frac80, 3),
            "ens_frac100_pct": round(frac100, 3),
        })
        print(f"==> ENSEMBLE [{orient} z={z0}]: p99.5={p995:.1f}, >80={frac80:.2f}%, >100={frac100:.2f}% -> {ov_path.name}", flush=True)

    (args.out_dir / "scan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Done scanning {args.zarr_dir.name} in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
