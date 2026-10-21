// Standalone bit-exact C++ port of ScrollPrize/villa vc_render_tifxyz
// (commit 41701aa0eaa1aa765523db3f9f972b5830abf549 / 0fb38c45b488e192566ea83ae36369cb6db92652)
// Implements QuadSurface::gen, grid_normal_int, prepareBaseAndDirs, and sampleTileSlicesImpl
// for rendering a 66-layer surface volume zarr from a .tifxyz mesh and 128^3 volume chunks.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

struct Vec3f {
    float v[3];
    float& operator[](int i) { return v[i]; }
    const float& operator[](int i) const { return v[i]; }
};

static inline int clampIndex(int v, int n) {
    return std::max(0, std::min(v, n - 1));
}

static inline bool loc_valid(const std::vector<Vec3f>& pts, int rows, int cols, int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) return false;
    const Vec3f& p = pts[r * cols + c];
    return (p[0] != -1.0f) && (!std::isnan(p[0]));
}

static void compute_grid_normals(
    const std::vector<Vec3f>& pts, int rows, int cols, std::vector<Vec3f>& normals)
{
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    normals.assign(rows * cols, Vec3f{{qnan, qnan, qnan}});
    for (int r = 1; r < rows - 1; ++r) {
        for (int c = 1; c < cols - 1; ++c) {
            if (!loc_valid(pts, rows, cols, r, c) ||
                !loc_valid(pts, rows, cols, r + 1, c) ||
                !loc_valid(pts, rows, cols, r - 1, c) ||
                !loc_valid(pts, rows, cols, r, c + 1) ||
                !loc_valid(pts, rows, cols, r, c - 1)) {
                continue;
            }
            const Vec3f& pr_p1 = pts[r * cols + (c + 1)];
            const Vec3f& pr_m1 = pts[r * cols + (c - 1)];
            const Vec3f& pc_p1 = pts[(r + 1) * cols + c];
            const Vec3f& pc_m1 = pts[(r - 1) * cols + c];
            float xv0 = pr_p1[0] - pr_m1[0];
            float xv1 = pr_p1[1] - pr_m1[1];
            float xv2 = pr_p1[2] - pr_m1[2];
            float yv0 = pc_p1[0] - pc_m1[0];
            float yv1 = pc_p1[1] - pc_m1[1];
            float yv2 = pc_p1[2] - pc_m1[2];
            float n0 = xv1 * yv2 - xv2 * yv1;
            float n1 = xv2 * yv0 - xv0 * yv2;
            float n2 = xv0 * yv1 - xv1 * yv0;
            float len2 = n0 * n0 + n1 * n1 + n2 * n2;
            if (len2 > 0.0f) {
                float inv = 1.0f / std::sqrt(len2);
                n0 *= inv;
                n1 *= inv;
                n2 *= inv;
                if (!std::isnan(n0) && !std::isnan(n1) && !std::isnan(n2)) {
                    normals[r * cols + c] = Vec3f{{n0, n1, n2}};
                }
            }
        }
    }
}

static void gen_tile(
    const std::vector<Vec3f>& pts,
    const std::vector<uint8_t>& valid_src,
    const std::vector<Vec3f>& normal_cache,
    int srcH, int srcW,
    float scale_x, float scale_y,
    float center_x, float center_y,
    int outW, int outH,
    float u0, float v0,
    std::vector<Vec3f>& base_out,
    std::vector<Vec3f>& dirs_out)
{
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    const float ul0 = (u0 + center_x) * scale_x;
    const float ul1 = (v0 + center_y) * scale_y;
    const int w_big = outW + 8;
    const int h_big = outH + 8;
    const double sx = static_cast<double>(scale_x);
    const double sy = static_cast<double>(scale_y);
    const double ox = static_cast<double>(ul0) - 4.0 * sx;
    const double oy = static_cast<double>(ul1) - 4.0 * sy;

    const float oxf = static_cast<float>(ox);
    const float oyf = static_cast<float>(oy);
    const float sxf = static_cast<float>(sx);
    const float syf = static_cast<float>(sy);

    base_out.assign(outH * outW, Vec3f{{qnan, qnan, qnan}});
    dirs_out.assign(outH * outW, Vec3f{{qnan, qnan, qnan}});

    const float sxmax = static_cast<float>(srcW - 1);
    const float symax = static_cast<float>(srcH - 1);

    for (int yy = 0; yy < outH; ++yy) {
        const int dy = yy + 4;
        // warpNearestConstU8 & warpNearestConstVec3f
        const int ny = static_cast<int>(std::lround(oyf + static_cast<float>(dy) * syf));
        // warpBilinearReplicateVec3f
        float fy = oyf + static_cast<float>(dy) * syf;
        fy = fy < 0.0f ? 0.0f : (fy > symax ? symax : fy);
        const int y0c = static_cast<int>(fy);
        const int y1c = std::min(y0c + 1, srcH - 1);
        const float wy = fy - static_cast<float>(y0c);
        const float iwy = 1.0f - wy;

        for (int xx = 0; xx < outW; ++xx) {
            const int dx = xx + 4;
            const int nx = static_cast<int>(std::lround(oxf + static_cast<float>(dx) * sxf));
            uint8_t vmask = 0;
            if (ny >= 0 && ny < srcH && nx >= 0 && nx < srcW) {
                vmask = valid_src[ny * srcW + nx];
            }
            if (!vmask) {
                continue;
            }
            float fx = oxf + static_cast<float>(dx) * sxf;
            fx = fx < 0.0f ? 0.0f : (fx > sxmax ? sxmax : fx);
            const int x0c = static_cast<int>(fx);
            const int x1c = std::min(x0c + 1, srcW - 1);
            const float wx = fx - static_cast<float>(x0c);
            const float iwx = 1.0f - wx;

            const Vec3f& p00 = pts[y0c * srcW + x0c];
            const Vec3f& p01 = pts[y0c * srcW + x1c];
            const Vec3f& p10 = pts[y1c * srcW + x0c];
            const Vec3f& p11 = pts[y1c * srcW + x1c];

            Vec3f c_pt;
            c_pt[0] = (p00[0] * iwx + p01[0] * wx) * iwy + (p10[0] * iwx + p11[0] * wx) * wy;
            c_pt[1] = (p00[1] * iwx + p01[1] * wx) * iwy + (p10[1] * iwx + p11[1] * wx) * wy;
            c_pt[2] = (p00[2] * iwx + p01[2] * wx) * iwy + (p10[2] * iwx + p11[2] * wx) * wy;

            Vec3f n_pt = (ny >= 0 && ny < srcH && nx >= 0 && nx < srcW)
                ? normal_cache[ny * srcW + nx]
                : Vec3f{{qnan, qnan, qnan}};

            if (!std::isnan(n_pt[0])) {
                const float L2 = n_pt[0] * n_pt[0] + n_pt[1] * n_pt[1] + n_pt[2] * n_pt[2];
                if (L2 > 0.0f) {
                    const float len = std::sqrt(L2);
                    n_pt[0] /= len;
                    n_pt[1] /= len;
                    n_pt[2] /= len;
                }
            }
            base_out[yy * outW + xx] = c_pt;
            dirs_out[yy * outW + xx] = n_pt;
        }
    }
}

static constexpr int VOL_Z = 17998;
static constexpr int VOL_Y = 8595;
static constexpr int VOL_X = 8595;
static constexpr int CHUNK = 128;

static inline bool inBounds(float vz, float vy, float vx) {
    return vz >= 0.0f && vy >= 0.0f && vx >= 0.0f &&
           vz < static_cast<float>(VOL_Z) &&
           vy < static_cast<float>(VOL_Y) &&
           vx < static_cast<float>(VOL_X);
}

struct ChunkHash {
    size_t operator()(uint64_t k) const noexcept { return std::hash<uint64_t>{}(k); }
};

static inline uint64_t packKey(int iz, int iy, int ix) {
    return (static_cast<uint64_t>(iz) << 32) |
           (static_cast<uint64_t>(iy) << 16) |
           static_cast<uint64_t>(ix);
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " --list-chunks <x.bin> <y.bin> <z.bin> <chunks_out.txt>\n"
                  << "  " << argv[0] << " --render <x.bin> <y.bin> <z.bin> <chunks_dir> <out_zarr_0_dir>\n";
        return 1;
    }
    std::string mode = argv[1];
    std::string x_path = argv[2];
    std::string y_path = argv[3];
    std::string z_path = argv[4];

    const int rows = 128;
    const int cols = 336;
    std::vector<float> xb(rows * cols), yb(rows * cols), zb(rows * cols);
    {
        std::ifstream fx(x_path, std::ios::binary);
        std::ifstream fy(y_path, std::ios::binary);
        std::ifstream fz(z_path, std::ios::binary);
        if (!fx || !fy || !fz) {
            std::cerr << "Failed to open x/y/z bin files\n";
            return 1;
        }
        fx.read(reinterpret_cast<char*>(xb.data()), xb.size() * sizeof(float));
        fy.read(reinterpret_cast<char*>(yb.data()), yb.size() * sizeof(float));
        fz.read(reinterpret_cast<char*>(zb.data()), zb.size() * sizeof(float));
    }

    const float qnan = std::numeric_limits<float>::quiet_NaN();
    std::vector<Vec3f> pts(rows * cols);
    std::vector<uint8_t> valid_src(rows * cols, 0);
    for (int i = 0; i < rows * cols; ++i) {
        pts[i] = Vec3f{{xb[i], yb[i], zb[i]}};
        if ((pts[i][0] != -1.0f || pts[i][1] != -1.0f || pts[i][2] != -1.0f) &&
            !std::isnan(pts[i][0]) && !std::isnan(pts[i][1]) && !std::isnan(pts[i][2])) {
            valid_src[i] = 255;
        }
        if (pts[i][0] == -1.0f) {
            pts[i] = Vec3f{{qnan, qnan, qnan}};
        }
    }

    std::vector<Vec3f> normal_cache;
    compute_grid_normals(pts, rows, cols, normal_cache);

    const int fullW = 6720;
    const int fullH = 2560;
    const float scale_x = 0.05f;
    const float scale_y = 0.05f;
    const float center_x = static_cast<float>(cols) / scale_x / 2.0f;
    const float center_y = static_cast<float>(rows) / scale_y / 2.0f;
    const int numSlices = 66;
    std::vector<float> offsets(numSlices);
    for (int zi = 0; zi < numSlices; ++zi) {
        offsets[zi] = static_cast<float>((zi - 32.5) * 1.0);
    }

    const int tilesX = (fullW + 127) / 128; // 53
    const int tilesY = (fullH + 127) / 128; // 20

    if (mode == "--list-chunks") {
        std::string out_txt = argv[5];
        constexpr int NZ = 145, NY = 70, NX = 70;
        std::vector<uint8_t> needed(NZ * NY * NX, 0);
        for (int ty = 0; ty < tilesY; ++ty) {
            int y0 = ty * 128;
            int dy = std::min(128, fullH - y0);
            for (int tx = 0; tx < tilesX; ++tx) {
                int x0 = tx * 128;
                int dx = std::min(128, fullW - x0);
                float u0 = -0.5f * (fullW - 1.0f) + static_cast<float>(x0);
                float v0 = -0.5f * (fullH - 1.0f) + static_cast<float>(y0);
                std::vector<Vec3f> base, dirs;
                gen_tile(pts, valid_src, normal_cache, rows, cols,
                         scale_x, scale_y, center_x, center_y,
                         dx, dy, u0, v0, base, dirs);
                for (int i = 0; i < dx * dy; ++i) {
                    const Vec3f& b = base[i];
                    const Vec3f& d = dirs[i];
                    if (!std::isfinite(b[0]) || !std::isfinite(b[1]) || !std::isfinite(b[2]) ||
                        !std::isfinite(d[0]) || !std::isfinite(d[1]) || !std::isfinite(d[2])) {
                        continue;
                    }
                    for (int zi = 0; zi < numSlices; ++zi) {
                        float off = offsets[zi];
                        float vx = b[0] + d[0] * off;
                        float vy = b[1] + d[1] * off;
                        float vz = b[2] + d[2] * off;
                        if (!inBounds(vz, vy, vx)) continue;
                        int iz0 = static_cast<int>(vz);
                        int iy0 = static_cast<int>(vy);
                        int ix0 = static_cast<int>(vx);
                        int cz0 = iz0 >> 7, cz1 = (iz0 + 1) >> 7;
                        int cy0 = iy0 >> 7, cy1 = (iy0 + 1) >> 7;
                        int cx0 = ix0 >> 7, cx1 = (ix0 + 1) >> 7;
                        needed[(cz0 * NY + cy0) * NX + cx0] = 1;
                        if (cz1 != cz0 || cy1 != cy0 || cx1 != cx0) {
                            for (int dz = 0; dz <= 1; ++dz)
                                for (int dyy = 0; dyy <= 1; ++dyy)
                                    for (int dxx = 0; dxx <= 1; ++dxx)
                                        needed[(((iz0 + dz) >> 7) * NY + ((iy0 + dyy) >> 7)) * NX + ((ix0 + dxx) >> 7)] = 1;
                        }
                    }
                }
            }
        }
        std::ofstream fout(out_txt);
        int count = 0;
        for (int iz = 0; iz < NZ; ++iz)
            for (int iy = 0; iy < NY; ++iy)
                for (int ix = 0; ix < NX; ++ix)
                    if (needed[(iz * NY + iy) * NX + ix]) {
                        fout << iz << " " << iy << " " << ix << "\n";
                        count++;
                    }
        std::cout << "Total unique 128^3 chunks needed: " << count << "\n";
        return 0;
    }

    if (mode == "--render") {
        if (argc < 7) {
            std::cerr << "Usage: " << argv[0] << " --render <x.bin> <y.bin> <z.bin> <chunks_dir> <out_zarr_0_dir>\n";
            return 1;
        }
        std::string chunks_dir = argv[5];
        std::string out_dir = argv[6];

        // Load all available chunk files from chunks_dir into memory with flat O(1) grid
        constexpr int NZ = 145, NY = 70, NX = 70;
        std::vector<std::vector<uint8_t>> chunk_storage;
        std::vector<const uint8_t*> chunk_grid(NZ * NY * NX, nullptr);
        const size_t chunk_bytes = 128ULL * 128ULL * 128ULL;
        std::vector<std::pair<std::tuple<int,int,int>, std::filesystem::path>> files_to_load;
        for (const auto& entry : std::filesystem::directory_iterator(chunks_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string fn = entry.path().filename().string();
            int iz, iy, ix;
            if (std::sscanf(fn.c_str(), "%d_%d_%d.bin", &iz, &iy, &ix) == 3) {
                if (entry.file_size() == chunk_bytes && iz >= 0 && iz < NZ && iy >= 0 && iy < NY && ix >= 0 && ix < NX) {
                    files_to_load.push_back({{iz, iy, ix}, entry.path()});
                }
            }
        }
        chunk_storage.resize(files_to_load.size());
        for (size_t i = 0; i < files_to_load.size(); ++i) {
            auto [iz, iy, ix] = files_to_load[i].first;
            chunk_storage[i].resize(chunk_bytes);
            std::ifstream f(files_to_load[i].second, std::ios::binary);
            f.read(reinterpret_cast<char*>(chunk_storage[i].data()), chunk_bytes);
            chunk_grid[(iz * NY + iy) * NX + ix] = chunk_storage[i].data();
        }
        std::cout << "Loaded " << chunk_storage.size() << " non-empty chunks from " << chunks_dir << "\n";

        auto getVoxel = [&](int iz, int iy, int ix) -> float {
            const uint8_t* p = chunk_grid[(((iz >> 7) * NY + (iy >> 7)) * NX) + (ix >> 7)];
            if (!p) return 0.0f;
            int lz = iz & 127;
            int ly = iy & 127;
            int lx = ix & 127;
            return static_cast<float>(p[(lz * 128 + ly) * 128 + lx]);
        };

        auto sampleTrilinear = [&](float vz, float vy, float vx) -> float {
            int iz = static_cast<int>(vz);
            int iy = static_cast<int>(vy);
            int ix = static_cast<int>(vx);
            float fz = vz - static_cast<float>(iz);
            float fy = vy - static_cast<float>(iy);
            float fx = vx - static_cast<float>(ix);

            // Fast path: all 8 corners inside the same chunk
            if (((iz & 127) < 127) && ((iy & 127) < 127) && ((ix & 127) < 127)) {
                const uint8_t* base_p = chunk_grid[(((iz >> 7) * NY + (iy >> 7)) * NX) + (ix >> 7)];
                if (!base_p) return 0.0f;
                const uint8_t* p = base_p + (((iz & 127) * 128 + (iy & 127)) * 128 + (ix & 127));
                constexpr int sX = 1;
                constexpr int sY = 128;
                constexpr int sZ = 128 * 128;
                float c000 = static_cast<float>(p[0]);
                float c100 = static_cast<float>(p[sX]);
                float c010 = static_cast<float>(p[sY]);
                float c110 = static_cast<float>(p[sY + sX]);
                float c001 = static_cast<float>(p[sZ]);
                float c101 = static_cast<float>(p[sZ + sX]);
                float c011 = static_cast<float>(p[sZ + sY]);
                float c111 = static_cast<float>(p[sZ + sY + sX]);
                float c00 = std::fma(fx, c100 - c000, c000);
                float c10 = std::fma(fx, c110 - c010, c010);
                float c01 = std::fma(fx, c101 - c001, c001);
                float c11 = std::fma(fx, c111 - c011, c011);
                float c0 = std::fma(fy, c10 - c00, c00);
                float c1 = std::fma(fy, c11 - c01, c01);
                return std::fma(fz, c1 - c0, c0);
            }

            float c000 = getVoxel(iz, iy, ix);
            float c100 = getVoxel(iz, iy, ix + 1);
            float c010 = getVoxel(iz, iy + 1, ix);
            float c110 = getVoxel(iz, iy + 1, ix + 1);
            float c001 = getVoxel(iz + 1, iy, ix);
            float c101 = getVoxel(iz + 1, iy, ix + 1);
            float c011 = getVoxel(iz + 1, iy + 1, ix);
            float c111 = getVoxel(iz + 1, iy + 1, ix + 1);
            float c00 = std::fma(fx, c100 - c000, c000);
            float c10 = std::fma(fx, c110 - c010, c010);
            float c01 = std::fma(fx, c101 - c001, c001);
            float c11 = std::fma(fx, c111 - c011, c011);
            float c0 = std::fma(fy, c10 - c00, c00);
            float c1 = std::fma(fy, c11 - c01, c01);
            return std::fma(fz, c1 - c0, c0);
        };

        const size_t out_chunk_bytes = static_cast<size_t>(numSlices) * 128ULL * 128ULL;
        int written_chunks = 0;

        for (int ty = 0; ty < tilesY; ++ty) {
            int y0 = ty * 128;
            int dy = std::min(128, fullH - y0);
            std::filesystem::create_directories(out_dir + "/0/" + std::to_string(ty));
            for (int tx = 0; tx < tilesX; ++tx) {
                int x0 = tx * 128;
                int dx = std::min(128, fullW - x0);
                float u0 = -0.5f * (fullW - 1.0f) + static_cast<float>(x0);
                float v0 = -0.5f * (fullH - 1.0f) + static_cast<float>(y0);
                std::vector<Vec3f> base, dirs;
                gen_tile(pts, valid_src, normal_cache, rows, cols,
                         scale_x, scale_y, center_x, center_y,
                         dx, dy, u0, v0, base, dirs);

                std::vector<uint8_t> chunkBuf(out_chunk_bytes, 0);
                bool non_zero = false;
                for (int yy = 0; yy < dy; ++yy) {
                    for (int xx = 0; xx < dx; ++xx) {
                        const Vec3f& b = base[yy * dx + xx];
                        const Vec3f& d = dirs[yy * dx + xx];
                        if (!std::isfinite(b[0]) || !std::isfinite(b[1]) || !std::isfinite(b[2]) ||
                            !std::isfinite(d[0]) || !std::isfinite(d[1]) || !std::isfinite(d[2])) {
                            continue;
                        }
                        for (int zi = 0; zi < numSlices; ++zi) {
                            float off = offsets[zi];
                            float vx = b[0] + d[0] * off;
                            float vy = b[1] + d[1] * off;
                            float vz = b[2] + d[2] * off;
                            if (inBounds(vz, vy, vx)) {
                                float v = sampleTrilinear(vz, vy, vx);
                                uint8_t u = static_cast<uint8_t>(v);
                                if (u != 0) non_zero = true;
                                chunkBuf[(zi * 128 + yy) * 128 + xx] = u;
                            }
                        }
                    }
                }
                if (non_zero) {
                    std::string cpath = out_dir + "/0/" + std::to_string(ty) + "/" + std::to_string(tx);
                    std::ofstream fout(cpath, std::ios::binary);
                    fout.write(reinterpret_cast<const char*>(chunkBuf.data()), out_chunk_bytes);
                    written_chunks++;
                }
            }
        }
        std::cout << "Rendered and wrote " << written_chunks << " non-empty zarr chunks to " << out_dir << "\n";
        return 0;
    }

    return 1;
}
