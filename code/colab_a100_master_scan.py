#!/usr/bin/env python3
import os, sys, json, time, glob, subprocess, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path

print("=== [Colab A100 Master Scanner] Setting up environment ===", flush=True)
subprocess.run(["pip", "install", "-q", "tifffile", "zarr>=3.0.0", "pynrrd", "imagecodecs", "opencv-python-headless", "safetensors", "huggingface_hub", "timm", "segmentation-models-pytorch", "einops", "dynamic-network-architectures"], check=True)
if not os.path.exists("/content/villa"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ScrollPrize/villa.git", "/content/villa"], check=True)

os.makedirs("/content/ckpt", exist_ok=True)
os.makedirs("/content/code", exist_ok=True)

with open("/content/code/render_tifxyz_any.cpp", "w") as f:
    f.write("// General-purpose bit-exact C++ port of ScrollPrize/villa vc_render_tifxyz\n// Accepts dynamic (rows, cols) mesh dimensions and (volZ, volY, volX) volume bounds.\n\n#include <algorithm>\n#include <cmath>\n#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <filesystem>\n#include <fstream>\n#include <iostream>\n#include <limits>\n#include <set>\n#include <string>\n#include <tuple>\n#include <unordered_map>\n#include <vector>\n\nstruct Vec3f {\n    float v[3];\n    float& operator[](int i) { return v[i]; }\n    const float& operator[](int i) const { return v[i]; }\n};\n\nstatic inline int clampIndex(int v, int n) {\n    return std::max(0, std::min(v, n - 1));\n}\n\nstatic inline bool loc_valid(const std::vector<Vec3f>& pts, int rows, int cols, int r, int c) {\n    if (r < 0 || r >= rows || c < 0 || c >= cols) return false;\n    const Vec3f& p = pts[r * cols + c];\n    return (p[0] > 0.0f) && (!std::isnan(p[0]));\n}\n\nstatic void compute_grid_normals(\n    const std::vector<Vec3f>& pts, int rows, int cols, std::vector<Vec3f>& normals)\n{\n    const float qnan = std::numeric_limits<float>::quiet_NaN();\n    normals.assign(rows * cols, Vec3f{{qnan, qnan, qnan}});\n    for (int r = 1; r < rows - 1; ++r) {\n        for (int c = 1; c < cols - 1; ++c) {\n            if (!loc_valid(pts, rows, cols, r, c) ||\n                !loc_valid(pts, rows, cols, r + 1, c) ||\n                !loc_valid(pts, rows, cols, r - 1, c) ||\n                !loc_valid(pts, rows, cols, r, c + 1) ||\n                !loc_valid(pts, rows, cols, r, c - 1)) {\n                continue;\n            }\n            const Vec3f& pr_p1 = pts[r * cols + (c + 1)];\n            const Vec3f& pr_m1 = pts[r * cols + (c - 1)];\n            const Vec3f& pc_p1 = pts[(r + 1) * cols + c];\n            const Vec3f& pc_m1 = pts[(r - 1) * cols + c];\n            float xv0 = pr_p1[0] - pr_m1[0];\n            float xv1 = pr_p1[1] - pr_m1[1];\n            float xv2 = pr_p1[2] - pr_m1[2];\n            float yv0 = pc_p1[0] - pc_m1[0];\n            float yv1 = pc_p1[1] - pc_m1[1];\n            float yv2 = pc_p1[2] - pc_m1[2];\n            float n0 = xv1 * yv2 - xv2 * yv1;\n            float n1 = xv2 * yv0 - xv0 * yv2;\n            float n2 = xv0 * yv1 - xv1 * yv0;\n            float len2 = n0 * n0 + n1 * n1 + n2 * n2;\n            if (len2 > 0.0f) {\n                float inv = 1.0f / std::sqrt(len2);\n                n0 *= inv;\n                n1 *= inv;\n                n2 *= inv;\n                if (!std::isnan(n0) && !std::isnan(n1) && !std::isnan(n2)) {\n                    normals[r * cols + c] = Vec3f{{n0, n1, n2}};\n                }\n            }\n        }\n    }\n}\n\nstatic void gen_tile(\n    const std::vector<Vec3f>& pts,\n    const std::vector<uint8_t>& valid_src,\n    const std::vector<Vec3f>& normal_cache,\n    int srcH, int srcW,\n    float scale_x, float scale_y,\n    float center_x, float center_y,\n    int outW, int outH,\n    float u0, float v0,\n    std::vector<Vec3f>& base_out,\n    std::vector<Vec3f>& dirs_out)\n{\n    const float qnan = std::numeric_limits<float>::quiet_NaN();\n    const float ul0 = (u0 + center_x) * scale_x;\n    const float ul1 = (v0 + center_y) * scale_y;\n    const double sx = static_cast<double>(scale_x);\n    const double sy = static_cast<double>(scale_y);\n    const double ox = static_cast<double>(ul0) - 4.0 * sx;\n    const double oy = static_cast<double>(ul1) - 4.0 * sy;\n\n    const float oxf = static_cast<float>(ox);\n    const float oyf = static_cast<float>(oy);\n    const float sxf = static_cast<float>(sx);\n    const float syf = static_cast<float>(sy);\n\n    base_out.assign(outH * outW, Vec3f{{qnan, qnan, qnan}});\n    dirs_out.assign(outH * outW, Vec3f{{qnan, qnan, qnan}});\n\n    const float sxmax = static_cast<float>(srcW - 1);\n    const float symax = static_cast<float>(srcH - 1);\n\n    for (int yy = 0; yy < outH; ++yy) {\n        const int dy = yy + 4;\n        const int ny = static_cast<int>(std::lround(oyf + static_cast<float>(dy) * syf));\n        float fy = oyf + static_cast<float>(dy) * syf;\n        fy = fy < 0.0f ? 0.0f : (fy > symax ? symax : fy);\n        const int y0c = static_cast<int>(fy);\n        const int y1c = std::min(y0c + 1, srcH - 1);\n        const float wy = fy - static_cast<float>(y0c);\n\n        for (int xx = 0; xx < outW; ++xx) {\n            const int dx = xx + 4;\n            const int nx = static_cast<int>(std::lround(oxf + static_cast<float>(dx) * sxf));\n            uint8_t mask_val = 0;\n            Vec3f nrm{{qnan, qnan, qnan}};\n            if (ny >= 0 && ny < srcH && nx >= 0 && nx < srcW) {\n                mask_val = valid_src[ny * srcW + nx];\n                nrm = normal_cache[ny * srcW + nx];\n            }\n            if (!mask_val) continue;\n\n            float fx = oxf + static_cast<float>(dx) * sxf;\n            fx = fx < 0.0f ? 0.0f : (fx > sxmax ? sxmax : fx);\n            const int x0c = static_cast<int>(fx);\n            const int x1c = std::min(x0c + 1, srcW - 1);\n            const float wx = fx - static_cast<float>(x0c);\n\n            const Vec3f& v00 = pts[y0c * srcW + x0c];\n            const Vec3f& v10 = pts[y0c * srcW + x1c];\n            const Vec3f& v01 = pts[y1c * srcW + x0c];\n            const Vec3f& v11 = pts[y1c * srcW + x1c];\n\n            Vec3f b;\n            for (int c = 0; c < 3; ++c) {\n                float c0 = v00[c] + wx * (v10[c] - v00[c]);\n                float c1 = v01[c] + wx * (v11[c] - v01[c]);\n                b[c] = c0 + wy * (c1 - c0);\n            }\n            base_out[yy * outW + xx] = b;\n            dirs_out[yy * outW + xx] = nrm;\n        }\n    }\n}\n\nint main(int argc, char** argv) {\n    if (argc < 11) {\n        std::cerr << \"Usage:\\n\"\n                  << \"  \" << argv[0] << \" --list-chunks <x.bin> <y.bin> <z.bin> <rows> <cols> <volZ> <volY> <volX> <chunks_out.txt>\\n\"\n                  << \"  \" << argv[0] << \" --render <x.bin> <y.bin> <z.bin> <rows> <cols> <volZ> <volY> <volX> <chunks_dir> <out_zarr_dir>\\n\";\n        return 1;\n    }\n    std::string mode = argv[1];\n    std::string x_path = argv[2];\n    std::string y_path = argv[3];\n    std::string z_path = argv[4];\n    const int rows = std::atoi(argv[5]);\n    const int cols = std::atoi(argv[6]);\n    const int VOL_Z = std::atoi(argv[7]);\n    const int VOL_Y = std::atoi(argv[8]);\n    const int VOL_X = std::atoi(argv[9]);\n\n    auto inBounds = [&](float vz, float vy, float vx) -> bool {\n        return vz >= 0.0f && vy >= 0.0f && vx >= 0.0f &&\n               vz < static_cast<float>(VOL_Z) &&\n               vy < static_cast<float>(VOL_Y) &&\n               vx < static_cast<float>(VOL_X);\n    };\n\n    std::vector<float> xb(rows * cols), yb(rows * cols), zb(rows * cols);\n    {\n        std::ifstream fx(x_path, std::ios::binary);\n        std::ifstream fy(y_path, std::ios::binary);\n        std::ifstream fz(z_path, std::ios::binary);\n        if (!fx || !fy || !fz) {\n            std::cerr << \"Failed to open x/y/z bin files\\n\";\n            return 1;\n        }\n        fx.read(reinterpret_cast<char*>(xb.data()), xb.size() * sizeof(float));\n        fy.read(reinterpret_cast<char*>(yb.data()), yb.size() * sizeof(float));\n        fz.read(reinterpret_cast<char*>(zb.data()), zb.size() * sizeof(float));\n    }\n\n    const float qnan = std::numeric_limits<float>::quiet_NaN();\n    std::vector<Vec3f> pts(rows * cols);\n    std::vector<uint8_t> valid_src(rows * cols, 0);\n    for (int i = 0; i < rows * cols; ++i) {\n        pts[i] = Vec3f{{xb[i], yb[i], zb[i]}};\n        if ((pts[i][0] > 0.0f && pts[i][1] > 0.0f && pts[i][2] > 0.0f) &&\n            !std::isnan(pts[i][0]) && !std::isnan(pts[i][1]) && !std::isnan(pts[i][2])) {\n            valid_src[i] = 255;\n        } else {\n            pts[i] = Vec3f{{qnan, qnan, qnan}};\n        }\n    }\n\n    std::vector<Vec3f> normal_cache;\n    compute_grid_normals(pts, rows, cols, normal_cache);\n\n    const float scale_x = 0.05f;\n    const float scale_y = 0.05f;\n    const int fullW = static_cast<int>(std::lround(static_cast<double>(cols) / scale_x));\n    const int fullH = static_cast<int>(std::lround(static_cast<double>(rows) / scale_y));\n    const float center_x = static_cast<float>(cols) / scale_x / 2.0f;\n    const float center_y = static_cast<float>(rows) / scale_y / 2.0f;\n    const int numSlices = 66;\n    std::vector<float> offsets(numSlices);\n    for (int zi = 0; zi < numSlices; ++zi) {\n        offsets[zi] = static_cast<float>((zi - 32.5) * 1.0);\n    }\n\n    const int tilesX = (fullW + 127) / 128;\n    const int tilesY = (fullH + 127) / 128;\n    const int NZ = (VOL_Z + 127) / 128 + 2;\n    const int NY = (VOL_Y + 127) / 128 + 2;\n    const int NX = (VOL_X + 127) / 128 + 2;\n\n    if (mode == \"--list-chunks\") {\n        std::string out_txt = argv[10];\n        std::vector<uint8_t> needed(static_cast<size_t>(NZ) * NY * NX, 0);\n        for (int ty = 0; ty < tilesY; ++ty) {\n            int y0 = ty * 128;\n            int dy = std::min(128, fullH - y0);\n            for (int tx = 0; tx < tilesX; ++tx) {\n                int x0 = tx * 128;\n                int dx = std::min(128, fullW - x0);\n                float u0 = -0.5f * (fullW - 1.0f) + static_cast<float>(x0);\n                float v0 = -0.5f * (fullH - 1.0f) + static_cast<float>(y0);\n                std::vector<Vec3f> base, dirs;\n                gen_tile(pts, valid_src, normal_cache, rows, cols,\n                         scale_x, scale_y, center_x, center_y,\n                         dx, dy, u0, v0, base, dirs);\n                for (int i = 0; i < dx * dy; ++i) {\n                    const Vec3f& b = base[i];\n                    const Vec3f& d = dirs[i];\n                    if (!std::isfinite(b[0]) || !std::isfinite(b[1]) || !std::isfinite(b[2]) ||\n                        !std::isfinite(d[0]) || !std::isfinite(d[1]) || !std::isfinite(d[2])) {\n                        continue;\n                    }\n                    for (int zi = 0; zi < numSlices; ++zi) {\n                        float off = offsets[zi];\n                        float vx = b[0] + d[0] * off;\n                        float vy = b[1] + d[1] * off;\n                        float vz = b[2] + d[2] * off;\n                        if (!inBounds(vz, vy, vx)) continue;\n                        int iz0 = static_cast<int>(vz);\n                        int iy0 = static_cast<int>(vy);\n                        int ix0 = static_cast<int>(vx);\n                        for (int dz = 0; dz <= 1; ++dz)\n                            for (int dyy = 0; dyy <= 1; ++dyy)\n                                for (int dxx = 0; dxx <= 1; ++dxx)\n                                    needed[(((iz0 + dz) >> 7) * NY + ((iy0 + dyy) >> 7)) * NX + ((ix0 + dxx) >> 7)] = 1;\n                    }\n                }\n            }\n        }\n        std::ofstream fout(out_txt);\n        int count = 0;\n        for (int iz = 0; iz < NZ; ++iz)\n            for (int iy = 0; iy < NY; ++iy)\n                for (int ix = 0; ix < NX; ++ix)\n                    if (needed[(iz * NY + iy) * NX + ix]) {\n                        fout << iz << \" \" << iy << \" \" << ix << \"\\n\";\n                        count++;\n                    }\n        std::cout << \"Total unique 128^3 chunks needed: \" << count << \"\\n\";\n        return 0;\n    }\n\n    if (mode == \"--render\") {\n        if (argc < 11) return 1;\n        std::string chunks_dir = argv[10];\n        std::string out_dir = argv[11];\n\n        std::vector<std::vector<uint8_t>> chunk_storage;\n        std::vector<const uint8_t*> chunk_grid(static_cast<size_t>(NZ) * NY * NX, nullptr);\n        const size_t chunk_bytes = 128ULL * 128ULL * 128ULL;\n        std::vector<std::pair<std::tuple<int,int,int>, std::filesystem::path>> files_to_load;\n        for (const auto& entry : std::filesystem::directory_iterator(chunks_dir)) {\n            if (!entry.is_regular_file()) continue;\n            std::string fn = entry.path().filename().string();\n            int iz, iy, ix;\n            if (std::sscanf(fn.c_str(), \"%d_%d_%d.bin\", &iz, &iy, &ix) == 3) {\n                if (entry.file_size() == chunk_bytes && iz >= 0 && iz < NZ && iy >= 0 && iy < NY && ix >= 0 && ix < NX) {\n                    files_to_load.push_back({{iz, iy, ix}, entry.path()});\n                }\n            }\n        }\n        chunk_storage.resize(files_to_load.size());\n        for (size_t i = 0; i < files_to_load.size(); ++i) {\n            auto [iz, iy, ix] = files_to_load[i].first;\n            chunk_storage[i].resize(chunk_bytes);\n            std::ifstream f(files_to_load[i].second, std::ios::binary);\n            f.read(reinterpret_cast<char*>(chunk_storage[i].data()), chunk_bytes);\n            chunk_grid[(iz * NY + iy) * NX + ix] = chunk_storage[i].data();\n        }\n        std::cout << \"Loaded \" << chunk_storage.size() << \" non-empty chunks from \" << chunks_dir << \"\\n\";\n\n        auto getVoxel = [&](int iz, int iy, int ix) -> float {\n            const uint8_t* p = chunk_grid[(((iz >> 7) * NY + (iy >> 7)) * NX) + (ix >> 7)];\n            if (!p) return 0.0f;\n            int lz = iz & 127;\n            int ly = iy & 127;\n            int lx = ix & 127;\n            return static_cast<float>(p[(lz * 128 + ly) * 128 + lx]);\n        };\n\n        auto sampleTrilinear = [&](float vz, float vy, float vx) -> float {\n            int iz = static_cast<int>(vz);\n            int iy = static_cast<int>(vy);\n            int ix = static_cast<int>(vx);\n            float fz = vz - static_cast<float>(iz);\n            float fy = vy - static_cast<float>(iy);\n            float fx = vx - static_cast<float>(ix);\n\n            if (((iz & 127) < 127) && ((iy & 127) < 127) && ((ix & 127) < 127)) {\n                const uint8_t* base_p = chunk_grid[(((iz >> 7) * NY + (iy >> 7)) * NX) + (ix >> 7)];\n                if (!base_p) return 0.0f;\n                const uint8_t* p = base_p + (((iz & 127) * 128 + (iy & 127)) * 128 + (ix & 127));\n                constexpr int sX = 1;\n                constexpr int sY = 128;\n                constexpr int sZ = 128 * 128;\n                float c000 = static_cast<float>(p[0]);\n                float c100 = static_cast<float>(p[sX]);\n                float c010 = static_cast<float>(p[sY]);\n                float c110 = static_cast<float>(p[sY + sX]);\n                float c001 = static_cast<float>(p[sZ]);\n                float c101 = static_cast<float>(p[sZ + sX]);\n                float c011 = static_cast<float>(p[sZ + sY]);\n                float c111 = static_cast<float>(p[sZ + sY + sX]);\n                float c00 = std::fma(fx, c100 - c000, c000);\n                float c10 = std::fma(fx, c110 - c010, c010);\n                float c01 = std::fma(fx, c101 - c001, c001);\n                float c11 = std::fma(fx, c111 - c011, c011);\n                float c0 = std::fma(fy, c10 - c00, c00);\n                float c1 = std::fma(fy, c11 - c01, c01);\n                return std::fma(fz, c1 - c0, c0);\n            }\n\n            float c000 = getVoxel(iz, iy, ix);\n            float c100 = getVoxel(iz, iy, ix + 1);\n            float c010 = getVoxel(iz, iy + 1, ix);\n            float c110 = getVoxel(iz, iy + 1, ix + 1);\n            float c001 = getVoxel(iz + 1, iy, ix);\n            float c101 = getVoxel(iz + 1, iy, ix + 1);\n            float c011 = getVoxel(iz + 1, iy + 1, ix);\n            float c111 = getVoxel(iz + 1, iy + 1, ix + 1);\n            float c00 = std::fma(fx, c100 - c000, c000);\n            float c10 = std::fma(fx, c110 - c010, c010);\n            float c01 = std::fma(fx, c101 - c001, c001);\n            float c11 = std::fma(fx, c111 - c011, c011);\n            float c0 = std::fma(fy, c10 - c00, c00);\n            float c1 = std::fma(fy, c11 - c01, c01);\n            return std::fma(fz, c1 - c0, c0);\n        };\n\n        const size_t out_chunk_bytes = static_cast<size_t>(numSlices) * 128ULL * 128ULL;\n        int written_chunks = 0;\n\n        for (int ty = 0; ty < tilesY; ++ty) {\n            int y0 = ty * 128;\n            int dy = std::min(128, fullH - y0);\n            std::filesystem::create_directories(out_dir + \"/0/0/\" + std::to_string(ty));\n            for (int tx = 0; tx < tilesX; ++tx) {\n                int x0 = tx * 128;\n                int dx = std::min(128, fullW - x0);\n                float u0 = -0.5f * (fullW - 1.0f) + static_cast<float>(x0);\n                float v0 = -0.5f * (fullH - 1.0f) + static_cast<float>(y0);\n                std::vector<Vec3f> base, dirs;\n                gen_tile(pts, valid_src, normal_cache, rows, cols,\n                         scale_x, scale_y, center_x, center_y,\n                         dx, dy, u0, v0, base, dirs);\n\n                std::vector<uint8_t> chunkBuf(out_chunk_bytes, 0);\n                bool non_zero = false;\n                for (int yy = 0; yy < dy; ++yy) {\n                    for (int xx = 0; xx < dx; ++xx) {\n                        const Vec3f& b = base[yy * dx + xx];\n                        const Vec3f& d = dirs[yy * dx + xx];\n                        if (!std::isfinite(b[0]) || !std::isfinite(b[1]) || !std::isfinite(b[2]) ||\n                            !std::isfinite(d[0]) || !std::isfinite(d[1]) || !std::isfinite(d[2])) {\n                            continue;\n                        }\n                        for (int zi = 0; zi < numSlices; ++zi) {\n                            float off = offsets[zi];\n                            float vx = b[0] + d[0] * off;\n                            float vy = b[1] + d[1] * off;\n                            float vz = b[2] + d[2] * off;\n                            if (inBounds(vz, vy, vx)) {\n                                float v = sampleTrilinear(vz, vy, vx);\n                                uint8_t u = static_cast<uint8_t>(v);\n                                if (u != 0) non_zero = true;\n                                chunkBuf[(zi * 128 + yy) * 128 + xx] = u;\n                            }\n                        }\n                    }\n                }\n                if (non_zero) {\n                    std::string cpath = out_dir + \"/0/0/\" + std::to_string(ty) + \"/\" + std::to_string(tx);\n                    std::ofstream fout(cpath, std::ios::binary);\n                    fout.write(reinterpret_cast<const char*>(chunkBuf.data()), out_chunk_bytes);\n                    written_chunks++;\n                }\n            }\n        }\n        std::cout << \"Rendered and wrote \" << written_chunks << \" non-empty zarr chunks (\" << fullH << \"x\" << fullW << \") to \" << out_dir << \"\\n\";\n        return 0;\n    }\n\n    return 1;\n}\n")
with open("/content/code/fetch_s3_chunks.py", "w") as f:
    f.write("#!/usr/bin/env python3\n\"\"\"Stream required 128x128x128 OME-Zarr chunks from the public Vesuvius Challenge S3 bucket.\"\"\"\nimport os\nimport sys\nimport time\nfrom concurrent.futures import ThreadPoolExecutor, as_completed\nimport requests\nfrom requests.adapters import HTTPAdapter\n\nCHUNK_BYTES = 128 * 128 * 128\n\n\ndef main():\n    if len(sys.argv) < 4:\n        print(f\"Usage: {sys.argv[0]} <base_url> <needed_chunks.txt> <out_dir> [workers=32]\")\n        sys.exit(1)\n    base_url = sys.argv[1].rstrip(\"/\")\n    list_path = sys.argv[2]\n    out_dir = sys.argv[3]\n    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 32\n    os.makedirs(out_dir, exist_ok=True)\n\n    chunks = []\n    with open(list_path) as f:\n        for line in f:\n            parts = line.strip().split()\n            if len(parts) == 3:\n                chunks.append((int(parts[0]), int(parts[1]), int(parts[2])))\n\n    todo = []\n    already = 0\n    for iz, iy, ix in chunks:\n        bin_path = os.path.join(out_dir, f\"{iz}_{iy}_{ix}.bin\")\n        miss_path = os.path.join(out_dir, f\"{iz}_{iy}_{ix}.404\")\n        if (os.path.exists(bin_path) and os.path.getsize(bin_path) == CHUNK_BYTES) or os.path.exists(miss_path):\n            already += 1\n        else:\n            todo.append((iz, iy, ix))\n\n    print(f\"Total chunks in list: {len(chunks)}, already cached: {already}, to fetch: {len(todo)}\", flush=True)\n    if not todo:\n        return\n\n    session = requests.Session()\n    adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers, max_retries=3)\n    session.mount(\"https://\", adapter)\n\n    def fetch_one(key):\n        iz, iy, ix = key\n        url = f\"{base_url}/{iz}/{iy}/{ix}\"\n        bin_path = os.path.join(out_dir, f\"{iz}_{iy}_{ix}.bin\")\n        miss_path = os.path.join(out_dir, f\"{iz}_{iy}_{ix}.404\")\n        for attempt in range(4):\n            try:\n                r = session.get(url, timeout=30)\n                if r.status_code == 200:\n                    data = r.content\n                    if len(data) != CHUNK_BYTES:\n                        raise RuntimeError(f\"Expected {CHUNK_BYTES} bytes, got {len(data)}\")\n                    tmp = bin_path + \".tmp\"\n                    with open(tmp, \"wb\") as fh:\n                        fh.write(data)\n                    os.replace(tmp, bin_path)\n                    return \"ok\"\n                elif r.status_code == 404:\n                    with open(miss_path, \"wb\") as fh:\n                        pass\n                    return \"404\"\n                else:\n                    time.sleep(0.5 * (attempt + 1))\n            except Exception as e:\n                if attempt == 3:\n                    raise e\n                time.sleep(0.5 * (attempt + 1))\n        raise RuntimeError(f\"Failed to fetch {url}\")\n\n    t0 = time.time()\n    n_ok = 0\n    n_404 = 0\n    with ThreadPoolExecutor(max_workers=workers) as ex:\n        futs = {ex.submit(fetch_one, k): k for k in todo}\n        for i, fut in enumerate(as_completed(futs), 1):\n            res = fut.result()\n            if res == \"ok\":\n                n_ok += 1\n            else:\n                n_404 += 1\n            if i % 50 == 0 or i == len(todo):\n                dt = time.time() - t0\n                mb = (n_ok * CHUNK_BYTES) / (1024 * 1024)\n                print(\n                    f\"  [{i}/{len(todo)}] ok={n_ok} 404={n_404} ({mb:.1f} MiB in {dt:.1f}s, {mb/max(dt,1e-3):.1f} MiB/s)\",\n                    flush=True,\n                )\n\n\nif __name__ == \"__main__\":\n    main()\n")

subprocess.run(["g++", "-O3", "-std=c++17", "/content/code/render_tifxyz_any.cpp", "-o", "/content/code/render_tifxyz_any"], check=True)
print("Compiled /content/code/render_tifxyz_any!", flush=True)

import torch
import numpy as np
import tifffile
import cv2
import zarr
from safetensors.torch import load_file

print(f"CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}", flush=True)

urls = [
    ("dense_native_016000.safetensors", "https://huggingface.co/Nieuwlaar/ink9um-dense-native/resolve/main/weights/dense_native_016000.safetensors"),
    ("train_dense_native.json", "https://huggingface.co/Nieuwlaar/ink9um-dense-native/resolve/main/training/configs/train_dense_native.json"),
    ("hybrid_3d2d-seed42_step-075000.pth", "https://huggingface.co/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed42/step-075000.pth"),
    ("hybrid_3d2d-seed43_step-075000.pth", "https://huggingface.co/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed43/step-075000.pth"),
    ("hecate_9.6um.pth", "https://huggingface.co/scrollprize/hecate/resolve/main/hecate_9.6um.pth"),
    ("hecate.py", "https://huggingface.co/scrollprize/hecate/resolve/main/hecate.py"),
]
for fn, url in urls:
    dst = f"/content/ckpt/{fn}"
    if not os.path.exists(dst):
        print(f"Downloading {fn} ...", flush=True)
        subprocess.run(["curl", "-fSL", "--retry", "4", "-o", dst, url], check=True)

if not os.path.exists("/content/ckpt/dense_native-016000.pth"):
    torch.save({"model": load_file("/content/ckpt/dense_native_016000.safetensors"), "config": json.load(open("/content/ckpt/train_dense_native.json")), "step": 16000}, "/content/ckpt/dense_native-016000.pth")

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
import argparse

print("Loading all 4 models onto A100 GPU ...", flush=True)
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
print("All 4 models loaded in VRAM!", flush=True)

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
    np.clip(out, 0.0, 1.0, out=out)
    return (out * 255.0).astype(np.uint8)

def eval_hecate_cuda(vol66: np.ndarray, reverse: bool) -> np.ndarray:
    H, W = vol66.shape[1], vol66.shape[2]
    out = np.zeros((H, W), dtype=np.uint8)
    hecate.predict(hecate_model, vol66, out, reverse=reverse, stride=64, batch_size=32, precision="bf16")
    return out

def make_overlay(ct_slice: np.ndarray, ink_u8: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    ct_eq = clahe.apply(ct_slice)
    rgb = np.stack([ct_eq, ct_eq, ct_eq], axis=-1).astype(np.float32)
    vals = ink_u8[valid_mask].astype(np.float32)
    if vals.size > 0:
        lo, hi = np.percentile(vals, [50.0, 99.5])
        alpha = np.clip((ink_u8.astype(np.float32) - lo) / max(hi - lo, 15.0), 0.0, 1.0)[..., None]
    else:
        alpha = np.zeros((ct_slice.shape[0], ct_slice.shape[1], 1), dtype=np.float32)
    alpha[~valid_mask] = 0.0
    red = np.zeros_like(rgb)
    red[..., 0] = 255.0
    red[..., 1] = 35.0
    red[..., 2] = 20.0
    comp = (rgb * (1.0 - 0.80 * alpha) + red * (0.80 * alpha)).clip(0, 255).astype(np.uint8)
    return comp

def create_zarr_metadata(zarr_dir: Path, num_slices: int, H: int, W: int):
    zarr_dir.mkdir(parents=True, exist_ok=True)
    (zarr_dir / ".zgroup").write_text('{"zarr_format": 2}\n')
    z0 = zarr_dir / "0"
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
    (z0 / ".zarray").write_text(json.dumps(zarray, indent=2) + "\n")

def scan_one_segment(scroll: str, seg_name: str, s3_seg_prefix: str, s3_vol_url: str, volZ: int, volY: int, volX: int, chunks_dir: str):
    t_seg = time.time()
    m_dir = Path(f"/content/meshes/{scroll}/{seg_name}")
    m_dir.mkdir(parents=True, exist_ok=True)
    for fn in ["meta.json", "x.tif", "y.tif", "z.tif"]:
        dst = m_dir / fn
        if not dst.exists():
            urllib.request.urlretrieve(f"https://vesuvius-challenge-open-data.s3.amazonaws.com/{s3_seg_prefix}/{fn}", str(dst))
    meta = json.load(open(m_dir / "meta.json"))
    x = tifffile.imread(str(m_dir / "x.tif")).astype(np.float32)
    y = tifffile.imread(str(m_dir / "y.tif")).astype(np.float32)
    z = tifffile.imread(str(m_dir / "z.tif")).astype(np.float32)
    rows, cols = x.shape
    x.tofile(str(m_dir / "x.bin"))
    y.tofile(str(m_dir / "y.bin"))
    z.tofile(str(m_dir / "z.bin"))
    chunks_txt = str(m_dir / "chunks.txt")
    subprocess.run(["/content/code/render_tifxyz_any", "--list-chunks", str(m_dir / "x.bin"), str(m_dir / "y.bin"), str(m_dir / "z.bin"), str(rows), str(cols), str(volZ), str(volY), str(volX), chunks_txt], check=True)
    subprocess.run(["python3", "/content/code/fetch_s3_chunks.py", s3_vol_url, chunks_txt, chunks_dir, "64"], check=True)

    zarr_dir = Path(f"/content/work/{scroll}/{seg_name}_66.zarr")
    subprocess.run(["/content/code/render_tifxyz_any", "--render", str(m_dir / "x.bin"), str(m_dir / "y.bin"), str(m_dir / "z.bin"), str(rows), str(cols), str(volZ), str(volY), str(volX), chunks_dir, str(zarr_dir)], check=True)
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
                vol66[:, y0:y0+dy, x0:x0+dx] = buf[:, :dy, :dx]
    valid_mask = vol66[32] > 0

    out_dir = Path(f"/content/results/{scroll}/{seg_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ct_mid = np.mean(vol66[29:37], axis=0).astype(np.uint8)
    cv2.imwrite(str(out_dir / "ct_mid.png"), ct_mid)

    seg_report = {"scroll": scroll, "segment": seg_name, "shape_yx": [H, W], "area_cm2": meta.get("area_cm2"), "orientations": {}}
    for orient, z0, rev_flag in [("reverse", 26, True), ("forward", 23, False)]:
        sub17 = vol66[z0:z0 + 17]
        sub17 = sub17[::-1].copy() if rev_flag else sub17.copy()
        m_dn = eval_villa_cuda(villa_models["dnative"], sub17)
        m_42 = eval_villa_cuda(villa_models["seed42"], sub17)
        m_43 = eval_villa_cuda(villa_models["seed43"], sub17)
        m_hc = eval_hecate_cuda(vol66, reverse=rev_flag)

        cv2.imwrite(str(out_dir / f"dnative_{orient}.png"), m_dn)
        cv2.imwrite(str(out_dir / f"seed42_{orient}.png"), m_42)
        cv2.imwrite(str(out_dir / f"seed43_{orient}.png"), m_43)
        cv2.imwrite(str(out_dir / f"hecate_{orient}.png"), m_hc)

        ens4 = np.clip(0.25 * m_dn.astype(np.float32) + 0.25 * m_42.astype(np.float32) + 0.25 * m_43.astype(np.float32) + 0.25 * m_hc.astype(np.float32), 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"ensemble4_{orient}.png"), ens4)
        ov4 = make_overlay(ct_mid, ens4, valid_mask)
        cv2.imwrite(str(out_dir / f"overlay4_{orient}.png"), cv2.cvtColor(ov4, cv2.COLOR_RGB2BGR))

        # Compute inter-model correlation between Villa 3-model average and 3D Hecate inside valid mask
        v_villa = (m_dn[valid_mask].astype(np.float32) + m_42[valid_mask].astype(np.float32) + m_43[valid_mask].astype(np.float32)) / 3.0
        v_hc = m_hc[valid_mask].astype(np.float32)
        r_vh = float(np.corrcoef(v_villa, v_hc)[0, 1]) if v_villa.size > 100 and np.std(v_villa) > 1e-3 and np.std(v_hc) > 1e-3 else 0.0
        vens = ens4[valid_mask]
        p995 = float(np.percentile(vens, 99.5)) if vens.size else 0.0
        p50 = float(np.percentile(vens, 50.0)) if vens.size else 0.0
        contrast = p995 - p50
        seg_report["orientations"][orient] = {
            "ens4_p995": round(p995, 2),
            "ens4_p50": round(p50, 2),
            "ens4_contrast": round(contrast, 2),
            "hecate_p995": round(float(np.percentile(v_hc, 99.5)) if v_hc.size else 0.0, 2),
            "r_villa_hecate": round(r_vh, 4),
        }
        print(f"  [{scroll}/{seg_name} {orient}] ens4 p99.5={p995:.1f} (contrast={contrast:.1f}), hecate p99.5={seg_report['orientations'][orient]['hecate_p995']:.1f}, r(villa,hecate)={r_vh:.4f}", flush=True)

    # Free rendered zarr to keep /content disk clean
    subprocess.run(["rm", "-rf", str(zarr_dir)], check=False)
    print(f"Finished {scroll}/{seg_name} in {time.time() - t_seg:.1f}s!", flush=True)
    return seg_report

all_reports = []

# 1. Scan all 6 wraps of PHerc0800 (8.640 um, 116 keV)
S3_VOL_0800 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr/0"
SEGS_0800 = [
    ("20251028213516", "PHerc0800/segments/20251028213516-auto_grown_20251028213516907/mesh/20251028213516-on-20250521135224-8.64um.tifxyz"),
    ("20251028220042", "PHerc0800/segments/20251028220042-auto_grown_20251028220042762/mesh/20251028220042-on-20250521135224-8.64um.tifxyz"),
    ("20251028220955", "PHerc0800/segments/20251028220955-auto_grown_20251028220955262/mesh/20251028220955-on-20250521135224-8.64um.tifxyz"),
    ("20251028222030", "PHerc0800/segments/20251028222030-auto_grown_20251028222030940/mesh/20251028222030-on-20250521135224-8.64um.tifxyz"),
    ("20251028225813", "PHerc0800/segments/20251028225813-auto_grown_20251028225813045/mesh/20251028225813-on-20250521135224-8.64um.tifxyz"),
    ("20251029010146", "PHerc0800/segments/20251029010146-auto_grown_20251029010146642/mesh/20251029010146-on-20250521135224-8.64um.tifxyz"),
]
for sname, spref in SEGS_0800:
    rep = scan_one_segment("PHerc0800", sname, spref, S3_VOL_0800, 24298, 9867, 9867, "/content/chunks/PHerc0800")
    all_reports.append(rep)
subprocess.run(["rm", "-rf", "/content/chunks/PHerc0800"], check=False)

# 2. Scan all 8 large segments of PHerc1203 (6.19 cm2 to 16.10 cm2)
S3_VOL_1203 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr/0"
SEGS_1203 = [
    "auto_grown_20250930000321811",  # 6.91 cm2
    "auto_grown_20250929222256117",  # 7.47 cm2
    "auto_grown_20251005230118636",  # 7.61 cm2
    "auto_grown_20251005221856743",  # 11.79 cm2
    "auto_grown_20250930104534929",  # 8.40 cm2
    "auto_grown_20251005230830030",  # 6.19 cm2
    "auto_grown_20251005PANORAMA_TARGET",  # 12.53 cm2
    "auto_grown_20251005230830031",  # 16.10 cm2
]
for sname in SEGS_1203:
    rep = scan_one_segment("PHerc1203", sname, f"PHerc1203/segments/raw/{sname}", S3_VOL_1203, 18977, 6844, 6844, "/content/chunks/PHerc1203")
    all_reports.append(rep)
subprocess.run(["rm", "-rf", "/content/chunks/PHerc1203"], check=False)

with open("/content/results/master_summary.json", "w") as f:
    json.dump(all_reports, f, indent=2)

subprocess.run(["tar", "-czf", "/content/unread_scan_results.tar.gz", "-C", "/content", "results"], check=True)
print("=== ALL 14 SEGMENTS ACROSS PHERC0800 & PHERC1203 SCANNED AND PACKAGED INTO /content/unread_scan_results.tar.gz ===", flush=True)
