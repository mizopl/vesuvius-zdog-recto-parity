#!/usr/bin/env python3
"""
Generator and launcher for parallel Google Colab GPU workers (Stage B + C: 19 unread scrolls).
Each worker:
  1. Uses its own isolated `/tmp/colab_w<k>/config.json` and session `vesuvius-w<k>`
  2. Uploads a compact tarball containing:
     - `render_tifxyz_any.cpp` + `fetch_s3_chunks.py`
     - `manifest_19_scrolls.json`
     - Pre-built `.tifxyz` surface meshes (`x.tif`, `y.tif`, `z.tif`, `meta.json`) for its assigned scrolls
  3. Runs 66-layer C++ surface rendering + 4-model ink detection (`dnative`, `seed42`, `seed43`, `hecate`)
     in both `reverse` and `forward` orientations
  4. Computes BOTH raw-average (`ens4`) AND 4-model equal-weight z-score ensemble (`ens_z`),
     plus the exact `pherc343-first-letters` gold dimmed-CT overlay (`submission_gold_{orient}.png`),
     and scans sliding 2 cm x 2 cm (`4.0 cm^2`) windows for peak cross-family agreement `max_win_r(villa, hecate)`
  5. Downloads `/content/worker_<k>_results.tar.gz` and stops the Colab VM in a `finally:` block.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
COLAB_BIN = "/opt/homebrew/bin/colab"

REMOTE_WORKER_CODE = r'''#!/usr/bin/env python3
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

print("=== STAGE B/C COLAB GPU WORKER STARTING ===", flush=True)
subprocess.run(["nvidia-smi"], check=False)

# 1. Install dependencies compatible with Colab Python 3.13 + compile C++ renderer
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "zarr>=3.0.0", "pynrrd", "imagecodecs", "tifffile", "safetensors", "huggingface_hub", "einops", "opencv-python-headless", "scipy"
], check=True)

with tarfile.open("/content/worker_input.tar.gz", "r:gz") as tar:
    tar.extractall("/content")

subprocess.run([
    "g++", "-O3", "-std=c++17", "-pthread",
    "/content/code/render_tifxyz_any.cpp", "-o", "/content/code/render_tifxyz_any"
], check=True)

if not os.path.exists("/content/villa"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ScrollPrize/villa.git", "/content/villa"], check=True)

os.makedirs("/content/ckpt", exist_ok=True)
import torch
import numpy as np
import tifffile
import cv2
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
        subprocess.run(["curl", "-fSL", "--retry", "4", "-o", dst, url], check=True)

if not os.path.exists("/content/ckpt/dense_native-016000.pth"):
    torch.save({
        "model": load_file("/content/ckpt/dense_native_016000.safetensors"),
        "config": json.load(open("/content/ckpt/train_dense_native.json")),
        "step": 16000
    }, "/content/ckpt/dense_native-016000.pth")

open("/content/villa/vesuvius/src/vesuvius/__init__.py", "w").write("")
sys.path.insert(0, "/content/villa/vesuvius/src")
sys.path.insert(0, "/content/ckpt")

from vesuvius.ink_detection.inference.infer import (
    _sliding_positions_1d,
    compute_importance_map_2d,
    configure_model,
    logits_to_probabilities,
    normalize_flat_patch,
)
import hecate

CKPT_DIR = Path("/content/ckpt")
villa_models = {}
for mname, fn in [
    ("dnative", "dense_native-016000.pth"),
    ("seed42", "hybrid_3d2d-seed42_step-075000.pth"),
    ("seed43", "hybrid_3d2d-seed43_step-075000.pth"),
]:
    cfg = configure_model(argparse.Namespace(checkpoint=CKPT_DIR / fn, amp_dtype="default"))
    p = int(cfg.patch_size)
    villa_models[mname] = (
        cfg.model.to("cuda").eval(),
        cfg.preprocessing,
        compute_importance_map_2d(patch_size=(p, p), mode="hann").numpy(),
        p,
    )

hecate_model = hecate.load_model(str(CKPT_DIR / "hecate_9.6um.pth"), "cuda")
print("All 4 models loaded on GPU!", flush=True)


def eval_villa_cuda(m_tuple, vol17: np.ndarray, bs: int = 64) -> np.ndarray:
    model, preproc, weight_map, p = m_tuple
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
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(coords), bs):
            batch_coords = coords[start:start + bs]
            batch_tensors = []
            for (y0, x0, vh, vw) in batch_coords:
                patch = np.zeros((17, p, p), dtype=vol17.dtype)
                patch[:, :vh, :vw] = vol17[:, y0:y0 + vh, x0:x0 + vw]
                normed = normalize_flat_patch(patch, preproc)
                batch_tensors.append(normed[None])
            x_batch = torch.from_numpy(np.stack(batch_tensors)).to("cuda", non_blocking=True)
            probs = logits_to_probabilities(model(x_batch), image_hw=(p, p)).float().cpu().numpy()[:, 0]
            for prob, (y0, x0, vh, vw) in zip(probs, batch_coords):
                w = weight_map[:vh, :vw]
                prob_sum[y0:y0 + vh, x0:x0 + vw] += prob[:vh, :vw] * w
                weight_sum[y0:y0 + vh, x0:x0 + vw] += w
    out = np.zeros((Vh, Vw), dtype=np.float32)
    np.divide(prob_sum, weight_sum, out=out, where=weight_sum > 1e-6)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def eval_hecate_cuda(vol66: np.ndarray, reverse: bool) -> np.ndarray:
    H, W = vol66.shape[1], vol66.shape[2]
    out = np.zeros((H, W), dtype=np.uint8)
    hecate.predict(hecate_model, vol66, out, reverse=reverse, stride=64, batch_size=32, precision="bf16")
    return out


def compute_zscore_ensemble_and_gold(ct_mid: np.ndarray, m_dn: np.ndarray, m_hc: np.ndarray,
                                     m_42: np.ndarray, m_43: np.ndarray, valid_mask: np.ndarray):
    """Exact 4-model equal-weight z-score ensemble + dimmed-CT gold overlay from PHerc. 343 pipeline."""
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
    return ens_u8, rgb


def max_window_correlation(villa_f: np.ndarray, hecate_f: np.ndarray, valid_mask: np.ndarray, um: float):
    """Find peak r(villa, hecate) across 2 cm x 2 cm (4 cm^2) windows."""
    win_px = int(round(20000.0 / um))  # 20,000 um = 2.0 cm
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
    ys = sorted(set(list(range(0, H - win_px + 1, 300)) + [H - win_px]))
    xs = sorted(set(list(range(0, W - win_px + 1, 300)) + [W - win_px]))
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


manifest = json.load(open("/content/manifest_19_scrolls.json"))
worker_cfg = json.load(open("/content/worker_assignment.json"))
assigned_scrolls = worker_cfg["scrolls"]
worker_id = worker_cfg["worker_id"]

all_reports = []
for scroll in assigned_scrolls:
    info = manifest[scroll]
    s3_scroll = info.get("s3_scroll", scroll)
    vol_name = info["volume"]
    um = float(info["um"])
    volZ, volY, volX = info["shape_zyx"]
    s3_vol_url = f"https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/{s3_scroll}/volumes/{vol_name}/0"
    chunks_dir = f"/content/chunks/{scroll}"
    os.makedirs(chunks_dir, exist_ok=True)

    scroll_mesh_dir = Path(f"/content/meshes/{scroll}")
    if not scroll_mesh_dir.exists():
        continue
    seg_dirs = sorted([d for d in scroll_mesh_dir.iterdir() if d.is_dir() and (d / "x.tif").exists()])

    for m_dir in seg_dirs:
        seg_name = m_dir.name
        t_seg = time.time()
        meta = json.load(open(m_dir / "meta.json"))
        x = tifffile.imread(str(m_dir / "x.tif")).astype(np.float32)
        y = tifffile.imread(str(m_dir / "y.tif")).astype(np.float32)
        z = tifffile.imread(str(m_dir / "z.tif")).astype(np.float32)
        rows, cols = x.shape
        x.tofile(str(m_dir / "x.bin"))
        y.tofile(str(m_dir / "y.bin"))
        z.tofile(str(m_dir / "z.bin"))
        chunks_txt = str(m_dir / "chunks.txt")
        subprocess.run([
            "/content/code/render_tifxyz_any", "--list-chunks",
            str(m_dir / "x.bin"), str(m_dir / "y.bin"), str(m_dir / "z.bin"),
            str(rows), str(cols), str(volZ), str(volY), str(volX), chunks_txt
        ], check=True)
        subprocess.run(["python3", "/content/code/fetch_s3_chunks.py", s3_vol_url, chunks_txt, chunks_dir, "64"], check=True)

        zarr_dir = Path(f"/content/work/{scroll}/{seg_name}_66.zarr")
        subprocess.run([
            "/content/code/render_tifxyz_any", "--render",
            str(m_dir / "x.bin"), str(m_dir / "y.bin"), str(m_dir / "z.bin"),
            str(rows), str(cols), str(volZ), str(volY), str(volX), chunks_dir, str(zarr_dir)
        ], check=True)

        H, W = rows * 20, cols * 20
        vol66 = np.zeros((66, H, W), dtype=np.uint8)
        tilesY, tilesX = (H + 127) // 128, (W + 127) // 128
        for ty in range(tilesY):
            y0, dy = ty * 128, min(128, H - ty * 128)
            for tx in range(tilesX):
                x0, dx = tx * 128, min(128, W - tx * 128)
                cp = zarr_dir / "0" / "0" / str(ty) / str(tx)
                if cp.exists():
                    buf = np.fromfile(str(cp), dtype=np.uint8).reshape(66, 128, 128)
                    vol66[:, y0:y0 + dy, x0:x0 + dx] = buf[:, :dy, :dx]
        valid_mask = vol66[32] > 0

        out_dir = Path(f"/content/results/{scroll}/{seg_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        ct_mid = vol66[34].copy()
        cv2.imwrite(str(out_dir / "ct_mid.png"), ct_mid)

        seg_report = {
            "scroll": scroll,
            "segment": seg_name,
            "um": um,
            "shape_yx": [H, W],
            "area_cm2": meta.get("area_cm2"),
            "valid_frac": round(float(valid_mask.mean()), 4),
            "orientations": {},
        }

        for orient, z0, rev_flag in [("reverse", 26, True), ("forward", 23, False)]:
            sub17 = vol66[z0:z0 + 17]
            sub17 = sub17[::-1].copy() if rev_flag else sub17.copy()
            m_dn = eval_villa_cuda(villa_models["dnative"], sub17)
            m_42 = eval_villa_cuda(villa_models["seed42"], sub17)
            m_43 = eval_villa_cuda(villa_models["seed43"], sub17)
            m_hc = eval_hecate_cuda(vol66, reverse=rev_flag)

            ens4 = np.clip(0.25 * m_dn.astype(np.float32) + 0.25 * m_42.astype(np.float32) + 0.25 * m_43.astype(np.float32) + 0.25 * m_hc.astype(np.float32), 0, 255).astype(np.uint8)
            ens_z_u8, gold_rgb = compute_zscore_ensemble_and_gold(ct_mid, m_dn, m_hc, m_42, m_43, valid_mask)

            cv2.imwrite(str(out_dir / f"ens_z_{orient}.png"), ens_z_u8)
            cv2.imwrite(str(out_dir / f"submission_gold_{orient}.png"), cv2.cvtColor(gold_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"dnative_{orient}.png"), m_dn)
            cv2.imwrite(str(out_dir / f"hecate_{orient}.png"), m_hc)

            villa_f = (m_dn.astype(np.float32) + m_42.astype(np.float32) + m_43.astype(np.float32)) / 3.0
            hecate_f = m_hc.astype(np.float32)
            if valid_mask.sum() > 1000:
                v_v = villa_f[valid_mask]
                v_h = hecate_f[valid_mask]
                r_vh = float(np.corrcoef(v_v, v_h)[0, 1]) if (v_v.std() > 1e-3 and v_h.std() > 1e-3) else 0.0
                p50, p99, p995 = [float(x) for x in np.percentile(ens4[valid_mask], [50.0, 99.0, 99.5])]
                hc_p995 = float(np.percentile(v_h, 99.5))
            else:
                r_vh, p50, p99, p995, hc_p995 = 0.0, 0.0, 0.0, 0.0, 0.0

            win_r, win_box = max_window_correlation(villa_f, hecate_f, valid_mask, um)
            seg_report["orientations"][orient] = {
                "ens4_p50": round(p50, 2),
                "ens4_p995": round(p995, 2),
                "ens4_contrast": round(p995 - p50, 2),
                "hecate_p995": round(hc_p995, 2),
                "r_villa_hecate": round(r_vh, 4),
                "max_2x2cm_r_villa_hecate": round(win_r, 4),
                "max_2x2cm_box_yxwh": win_box,
            }
            print(f"  [{scroll}/{seg_name} {orient}] area={meta.get('area_cm2')}cm2 ens4_p99.5={p995:.1f} (c={p995-p50:.1f}) hecate_p99.5={hc_p995:.1f} r_all={r_vh:.4f} r_2x2cm={win_r:.4f}", flush=True)

        (out_dir / "metrics.json").write_text(json.dumps(seg_report, indent=2) + "\n")
        all_reports.append(seg_report)
        shutil.rmtree(zarr_dir, ignore_errors=True)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Finished {scroll}/{seg_name} in {time.time() - t_seg:.1f}s!", flush=True)

    # Free scroll's cached S3 chunks after finishing all wraps of this scroll so disk stays clean
    shutil.rmtree(chunks_dir, ignore_errors=True)

Path("/content/results/summary_metrics.json").write_text(json.dumps(all_reports, indent=2) + "\n")
subprocess.run(["tar", "-czf", f"/content/worker_{worker_id}_results.tar.gz", "-C", "/content/results", "."], check=True)
print(f"=== WORKER {worker_id} COMPLETE! ===", flush=True)
'''


def run_worker(worker_id: int, scrolls: list[str], gpu: str = "A100"):
    cfg_dir = Path(f"/tmp/colab_w{worker_id}")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    src_cfg = Path.home() / ".config" / "colab-cli"
    for fn in ["token.json", "config.json", "settings.json"]:
        if (src_cfg / fn).exists():
            shutil.copy2(src_cfg / fn, cfg_dir / fn)

    # Create input tarball for this worker
    tar_path = cfg_dir / f"worker_{worker_id}_input.tar.gz"
    assign_path = cfg_dir / "worker_assignment.json"
    assign_path.write_text(json.dumps({"worker_id": worker_id, "scrolls": scrolls}, indent=2) + "\n")

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(BASE_DIR / "code" / "render_tifxyz_any.cpp", arcname="code/render_tifxyz_any.cpp")
        tar.add(BASE_DIR / "code" / "fetch_s3_chunks.py", arcname="code/fetch_s3_chunks.py")
        tar.add(BASE_DIR / "manifest_19_scrolls.json", arcname="manifest_19_scrolls.json")
        tar.add(assign_path, arcname="worker_assignment.json")
        for sc in scrolls:
            sc_mesh_dir = BASE_DIR / "meshes" / sc
            if sc_mesh_dir.exists():
                tar.add(sc_mesh_dir, arcname=f"meshes/{sc}")

    script_path = cfg_dir / f"remote_worker_{worker_id}.py"
    script_path.write_text(REMOTE_WORKER_CODE)

    out_tar = BASE_DIR / f"worker_{worker_id}_results.tar.gz"
    session_name = f"vesuvius-w{worker_id}"
    sessions_file = cfg_dir / "sessions.json"
    if not sessions_file.exists():
        sessions_file.write_text("{}\n")
    base_cmd = [COLAB_BIN, "--config", str(sessions_file)]

    try:
        res = subprocess.run(base_cmd + ["new", "-s", session_name, "--gpu", gpu])
        if res.returncode != 0 and gpu != "L4":
            print(f"[Worker {worker_id}] {gpu} allocation returned {res.returncode}, falling back to L4 ...", flush=True)
            subprocess.run(base_cmd + ["new", "-s", session_name, "--gpu", "L4"], check=True)
        elif res.returncode != 0:
            raise RuntimeError(f"Failed to allocate Colab GPU for {session_name}")
        subprocess.run(base_cmd + ["upload", "-s", session_name, str(tar_path), "/content/worker_input.tar.gz"], check=True)
        subprocess.run(base_cmd + ["exec", "--timeout", "36000", "-s", session_name, "-f", str(script_path)], check=False)
        subprocess.run(base_cmd + ["download", "-s", session_name, f"/content/worker_{worker_id}_results.tar.gz", str(out_tar)], check=True)
    finally:
        subprocess.run(base_cmd + ["stop", "-s", session_name], check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--gpu", default="A100")
    ap.add_argument("scrolls", nargs="+")
    args = ap.parse_args()
    run_worker(args.worker_id, args.scrolls, args.gpu)
