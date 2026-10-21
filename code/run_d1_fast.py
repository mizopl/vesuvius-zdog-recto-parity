#!/usr/bin/env python3
"""Fast, exact in-memory runner for the 4 ink models at d = +1 (reverse / reading side)
on both plain and padded renders, with exact tile-phase merging (merge_tile_phase.py).
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from torch.nn.utils import fuse_conv_bn_eval
import zarr

BASE = Path('/workspace/v3r1t4s/runs/2026-10-19-0910')
sys.path.insert(0, str(BASE / 'data/villa/vesuvius/src'))
sys.path.insert(0, str(BASE / 'work/ckpt'))

from vesuvius.ink_detection.inference.infer import (
    _sliding_positions_1d,
    compute_importance_map_2d,
    configure_model,
    logits_to_probabilities,
    normalize_flat_patch,
)
import hecate

N = 'concat_w047-w048_R5B2_z9500-11000'
CKPT_DIR = BASE / 'work/ckpt'
MAPS_PLAIN = BASE / 'work/maps_plain'
MAPS_PADDED = BASE / 'work/maps_padded'
MAPS_MERGED = BASE / 'work/maps'


def run_villa_models(vol_plain_rev: np.ndarray, is47: np.ndarray, pad_rows: int = 40, pad_cols: int = 200):
    H, W = is47.shape
    Hp, Wp = H + pad_rows, W + pad_cols
    vol_pad_rev = np.zeros((17, Hp, Wp), dtype=np.uint8)
    vol_pad_rev[:, pad_rows:, pad_cols:] = vol_plain_rev
    is47_pad = np.zeros((Hp, Wp), dtype=bool)
    is47_pad[pad_rows:, pad_cols:] = is47

    models = [
        ('dnative', CKPT_DIR / 'dense_native-016000.pth'),
        ('hybrid_3d2d-seed42', CKPT_DIR / 'hybrid_3d2d-seed42_step-075000.pth'),
        ('hybrid_3d2d-seed43', CKPT_DIR / 'hybrid_3d2d-seed43_step-075000.pth'),
    ]

    for mname, ckpt_path in models:
        out_merged = MAPS_MERGED / f'{N}__{mname}__L66s26_reverse.tif'
        out_plain = MAPS_PLAIN / f'{N}_66__{mname}__L66s26_reverse.tif'
        out_padded = MAPS_PADDED / f'{N}_66_padded__{mname}__L66s26_reverse.tif'
        if out_merged.exists() and out_merged.stat().st_size > 0:
            print(f'[skip] {out_merged.name} already exists', flush=True)
            continue

        t0 = time.time()
        cfg = configure_model(argparse.Namespace(checkpoint=ckpt_path, amp_dtype='default'))
        model = cfg.model.to('mps').eval()
        p = int(cfg.patch_size)
        stride = p // 2
        preproc = cfg.preprocessing
        weight_map = compute_importance_map_2d(patch_size=(p, p), mode='hann').numpy()
        bs = 16

        def eval_pass(vol: np.ndarray, need_mask: np.ndarray, label: str) -> np.ndarray:
            Vh, Vw = vol.shape[1], vol.shape[2]
            ys = _sliding_positions_1d(Vh, p, stride)
            xs = _sliding_positions_1d(Vw, p, stride)
            coords = []
            for y0 in ys:
                vh = min(p, Vh - y0)
                for x0 in xs:
                    vw = min(p, Vw - x0)
                    if not need_mask[y0:y0 + vh, x0:x0 + vw].any():
                        continue
                    sub = vol[:, y0:y0 + vh, x0:x0 + vw]
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
                        patch = np.zeros((17, p, p), dtype=vol.dtype)
                        patch[:, :vh, :vw] = vol[:, y0:y0 + vh, x0:x0 + vw]
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
            out *= 255.0
            print(f'  {mname} [{label}]: {len(coords)} patches in {time.time() - t0:.1f}s', flush=True)
            return out.astype(np.uint8)

        ma = eval_pass(vol_plain_rev, ~is47, 'plain')
        mp = eval_pass(vol_pad_rev, is47_pad, 'padded')
        tifffile.imwrite(out_plain, ma, compression='lzw')
        tifffile.imwrite(out_padded, mp, compression='lzw')
        mp_crop = mp[pad_rows:pad_rows + H, pad_cols:pad_cols + W]
        merged = np.where(is47, mp_crop, ma).astype(np.uint8)
        tifffile.imwrite(out_merged, merged, compression='lzw')
        print(f'[done] {out_merged.name} ({time.time() - t0:.1f}s total)', flush=True)
        del model
        torch.mps.empty_cache()


def run_hecate_model(is47: np.ndarray, pad_rows: int = 40, pad_cols: int = 200):
    out_merged = MAPS_MERGED / f'{N}__hecate__L66zc33.50_reverse.png'
    out_plain = MAPS_PLAIN / f'{N}_66__hecate__L66zc33.50_reverse.png'
    out_padded = MAPS_PADDED / f'{N}_66_padded__hecate__L66zc33.50_reverse.png'
    if out_merged.exists() and out_merged.stat().st_size > 0:
        print(f'[skip] {out_merged.name} already exists', flush=True)
        return

    t0 = time.time()
    v_plain = np.asarray(zarr.open(str(BASE / 'work/tmp_hec/prep_plain.zarr'), mode='r')['0'])[::-1].copy()
    v_pad = np.asarray(zarr.open(str(BASE / 'work/tmp_hec/prep_padded.zarr'), mode='r')['0'])[::-1].copy()

    H, W = v_plain.shape[1], v_plain.shape[2]
    Hp, Wp = v_pad.shape[1], v_pad.shape[2]
    dr, dc = Hp - H, Wp - W
    m47 = cv2.resize(is47.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    m47_pad = np.zeros((Hp, Wp), dtype=bool)
    m47_pad[dr:, dc:] = m47

    # Load Hecate, fuse Conv3d+BatchNorm3d in float32, and run backbone in float16 on MPS
    m32 = hecate.load_model(str(CKPT_DIR / 'hecate_9.6um.pth'), 'cpu')
    mf = copy.deepcopy(m32)
    del m32
    mf.canonical.backbone.conv1 = fuse_conv_bn_eval(mf.canonical.backbone.conv1, mf.canonical.backbone.bn1)
    mf.canonical.backbone.bn1 = torch.nn.Identity()
    for layer in [
        mf.canonical.backbone.layer1,
        mf.canonical.backbone.layer2,
        mf.canonical.backbone.layer3,
        mf.canonical.backbone.layer4,
    ]:
        for b in layer:
            b.conv1 = fuse_conv_bn_eval(b.conv1, b.bn1)
            b.bn1 = torch.nn.Identity()
            b.conv2 = fuse_conv_bn_eval(b.conv2, b.bn2)
            b.bn2 = torch.nn.Identity()
            b.conv3 = fuse_conv_bn_eval(b.conv3, b.bn3)
            b.bn3 = torch.nn.Identity()
            if b.downsample is not None:
                b.downsample = fuse_conv_bn_eval(b.downsample[0], b.downsample[1])

    mf.canonical.backbone.to(dtype=torch.float16)
    mf.canonical.normalization.to(dtype=torch.float16)
    mf = mf.to('mps').eval()

    def encode_fp16(x):
        if x.ndim == 4:
            x = x[:, None]
        x = mf.canonical.normalization(x.to(torch.float16))
        return [f.float() for f in mf.canonical.backbone(x)]

    mf.canonical.encode = encode_fp16

    d, p, _ = mf.patch_size
    stride = p
    win = torch.ones((p, p), dtype=torch.float32)
    window = win.numpy()
    win_mps = win.to('mps')
    bs = 4

    def eval_hecate(vol: np.ndarray, need_mask: np.ndarray, label: str) -> np.ndarray:
        Vh, Vw = vol.shape[1], vol.shape[2]
        origins = [(y, x) for y in hecate.axis_starts(Vh, p, stride) for x in hecate.axis_starts(Vw, p, stride)]
        numerator = np.zeros((Vh, Vw), dtype=np.float32)
        denominator = np.zeros((Vh, Vw), dtype=np.float32)
        active_coords = []
        for y, x in origins:
            ph, pw = min(p, Vh - y), min(p, Vw - x)
            denominator[y:y + ph, x:x + pw] += window[:ph, :pw]
            if not need_mask[y:y + ph, x:x + pw].any():
                continue
            raw = vol[:, y:y + ph, x:x + pw]
            if not raw.any():
                continue
            active_coords.append((y, x, ph, pw))

        print(f'  hecate [{label}]: {len(active_coords)} active needed patches (of {len(origins)} origins)', flush=True)
        t_pass = time.time()
        with torch.inference_mode():
            for start in range(0, len(active_coords), bs):
                batch = active_coords[start:start + bs]
                patches = []
                for y, x, ph, pw in batch:
                    patch = np.zeros((d, p, p), dtype=np.uint8)
                    patch[:, :ph, :pw] = vol[:, y:y + ph, x:x + pw]
                    patches.append(patch)
                stacked = np.stack(patches)
                raw_t = torch.from_numpy(stacked).to('mps')
                support = raw_t.any(1)
                normed = stacked.astype(np.float32) / mf.divisor
                image = torch.from_numpy(normed[:, None]).to('mps')
                valid = support[:, None, None].expand(-1, 1, d, -1, -1)
                logits = mf(image, valid)
                probability = logits.float().sigmoid()[:, 0]
                weighted = (probability * support * win_mps).cpu().numpy()
                for pred, (y, x, ph, pw) in zip(weighted, batch):
                    numerator[y:y + ph, x:x + pw] += pred[:ph, :pw]
                done = min(start + bs, len(active_coords))
                if start == 0 or done == len(active_coords) or (start // bs) % 50 == 0:
                    el = time.time() - t_pass
                    rate = done / max(el, 1e-3)
                    eta = (len(active_coords) - done) / max(rate, 1e-3)
                    print(f'    {label}: {done}/{len(active_coords)} ({rate:.1f} p/s, ETA {eta:.0f}s)', flush=True)

        prob = numerator / denominator
        values = np.rint(prob.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        return values

    ma = eval_hecate(v_plain, ~m47, 'plain')
    mp = eval_hecate(v_pad, m47_pad, 'padded')
    hecate.write_png(ma, out_plain)
    hecate.write_png(mp, out_padded)
    mp_crop = mp[dr:, dc:]
    merged = np.where(m47, mp_crop, ma).astype(np.uint8)
    hecate.write_png(merged, out_merged)
    meta = {
        'checkpoint': 'hecate_9.6um.pth',
        'sampling_um': 9.6,
        'shape_yx': [H, W],
        'reverse_z': True,
        'central_depth': 16,
        'precision': 'fp32',
        'stride': 32,
        'probabilities': 'uint8 = round(255 * probability); no min-max scaling',
    }
    out_merged.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'[done] {out_merged.name} ({time.time() - t0:.1f}s total)', flush=True)


def main():
    for d in [MAPS_PLAIN, MAPS_PADDED, MAPS_MERGED]:
        d.mkdir(parents=True, exist_ok=True)

    source = np.load(str(BASE / f'work/mesh/{N}_node_source.npz'))['source']
    is47 = np.repeat(np.repeat((source == 1) | (source == 4), 20, 0), 20, 1)

    villa_done = all(
        (MAPS_MERGED / f'{N}__{m}__L66s26_reverse.tif').exists()
        for m in ('dnative', 'hybrid_3d2d-seed42', 'hybrid_3d2d-seed43')
    )
    if not villa_done:
        print('Loading layers 26..43 (reverse) from 66-layer surface volume ...', flush=True)
        st = zarr.open(str(BASE / f'work/{N}_66.zarr'), mode='r')['0']
        vol_plain_rev = np.asarray(st[26:43])[::-1].copy()
        run_villa_models(vol_plain_rev, is47)
        del vol_plain_rev
    else:
        print('[skip] all 3 villa models already computed', flush=True)
    run_hecate_model(is47)


if __name__ == '__main__':
    main()
