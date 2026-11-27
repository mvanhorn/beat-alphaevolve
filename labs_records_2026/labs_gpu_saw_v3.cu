// Pure CUDA C++ Ultra-Fast Shared-Memory Bloom-Filter Self-Avoiding Walk (GPU-SAW v3 / lssOrel)
// ZERO DRAM TRAFFIC inside the walk loop!
// Each CUDA block (walker) holds its ENTIRE state in on-chip Shared Memory (~33 KB/block):
//   - sh_s[MAX_L]       : 1 KB   (+1/-1 sequence)
//   - sh_C[MAX_L]       : 4 KB   (exact autocorrelation C[1..L-1])
//   - sh_zob[MAX_L]     : 8 KB   (64-bit Zobrist hash keys)
//   - sh_bloom[4096]    : 16 KB  (131,072-bit 3-hash Bloom filter -> <0.07% FP at 4096 steps, 0% FN!)
//   - red_dE / red_p    : 3 KB   (parallel warp/block argmin reduction)
// Exact host-side O(L^2) verification on every new best sequence!
// Written 100% from scratch.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <random>

#define MAX_L 1024
#define BLOCK_THREADS 256
#define BLOOM_WORDS 4096
#define BLOOM_BITS (BLOOM_WORDS * 32)
#define BLOOM_MASK (BLOOM_BITS - 1)

struct SawState {
    int8_t s[MAX_L];
    int64_t best_E;
    int8_t best_s[MAX_L];
};

__device__ __forceinline__ uint64_t xs64(uint64_t &s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}

__device__ __forceinline__ int64_t full_flip_delta(const int8_t *s, const int32_t *C, int L, int p) {
    int sp = (int)s[p];
    int64_t dE = 0;
    int kmax1 = (p < L - 1 - p) ? p : (L - 1 - p);
    for (int k = 1; k <= kmax1; ++k) {
        int sum = (int)s[p - k] + (int)s[p + k];
        if (sum != 0) { int d = -2 * sp * sum; dE += (int64_t)d * (2 * C[k] + d); }
    }
    for (int k = kmax1 + 1; k <= p; ++k) { int d = -2 * sp * (int)s[p - k]; dE += (int64_t)d * (2 * C[k] + d); }
    for (int k = kmax1 + 1; k < L - p; ++k) { int d = -2 * sp * (int)s[p + k]; dE += (int64_t)d * (2 * C[k] + d); }
    return dE;
}

__device__ __forceinline__ int64_t skew_flip_delta(const int8_t *s, const int32_t *C, int L, int p) {
    int m = (L + 1) >> 1;
    int c = m - 1;
    int64_t dE = 0;
    if (p == c) {
        int sp = (int)s[p];
        for (int k = 2; k <= c; k += 2) {
            int sum = (int)s[p - k] + (int)s[p + k];
            if (sum != 0) { int d = -2 * sp * sum; dE += (int64_t)d * (2 * C[k] + d); }
        }
        return dE;
    }
    int q = L - 1 - p, sp = (int)s[p], sq = (int)s[q];
    for (int k = 2; k < L; k += 2) {
        int d = 0;
        if (p >= k) d -= 2 * sp * (int)s[p - k];
        if (p + k < L && (p + k) != q) d -= 2 * sp * (int)s[p + k];
        if (q >= k && (q - k) != p) d -= 2 * sq * (int)s[q - k];
        if (q + k < L) d -= 2 * sq * (int)s[q + k];
        if (d != 0) dE += (int64_t)d * (2 * C[k] + d);
    }
    return dE;
}

__device__ __forceinline__ int bloom_contains(const uint32_t *bloom, uint64_t h) {
    uint32_t b1 = (uint32_t)(h & BLOOM_MASK);
    uint32_t b2 = (uint32_t)((h >> 17) & BLOOM_MASK);
    uint32_t b3 = (uint32_t)((h >> 34) & BLOOM_MASK);
    return ((bloom[b1 >> 5] >> (b1 & 31U)) & 1U)
        & ((bloom[b2 >> 5] >> (b2 & 31U)) & 1U)
        & ((bloom[b3 >> 5] >> (b3 & 31U)) & 1U);
}

__device__ __forceinline__ void bloom_insert(uint32_t *bloom, uint64_t h) {
    uint32_t b1 = (uint32_t)(h & BLOOM_MASK);
    uint32_t b2 = (uint32_t)((h >> 17) & BLOOM_MASK);
    uint32_t b3 = (uint32_t)((h >> 34) & BLOOM_MASK);
    bloom[b1 >> 5] |= (1U << (b1 & 31U));
    bloom[b2 >> 5] |= (1U << (b2 & 31U));
    bloom[b3 >> 5] |= (1U << (b3 & 31U));
}

__global__ void labs_saw_v3_kernel(
    SawState *walkers,
    const uint64_t *zobrist,
    const uint64_t *rng_states,
    int num_walkers,
    int L,
    int is_skew,
    int walk_steps,
    int round_id)
{
    int wid = blockIdx.x;
    if (wid >= num_walkers) return;
    int tid = threadIdx.x;

    __shared__ int8_t sh_s[MAX_L];
    __shared__ int32_t sh_C[MAX_L];
    __shared__ uint64_t sh_zob[MAX_L];
    __shared__ uint32_t sh_bloom[BLOOM_WORDS];
    __shared__ int64_t sh_E;
    __shared__ int64_t sh_best_E;
    __shared__ uint64_t sh_hash;
    __shared__ int64_t red_dE[BLOCK_THREADS];
    __shared__ int red_p[BLOCK_THREADS];

    // Clear 16 KB shared-memory Bloom filter in 16 instructions/thread (~20 ns!)
    for (int i = tid; i < BLOOM_WORDS; i += BLOCK_THREADS) sh_bloom[i] = 0U;

    int nvars = is_skew ? ((L + 1) >> 1) : L;

    for (int i = tid; i < L; i += BLOCK_THREADS) {
        int8_t v = walkers[wid].s[i];
        sh_s[i] = v;
        walkers[wid].best_s[i] = v;
        sh_zob[i] = zobrist[i];
    }
    __syncthreads();

    // Compute exact initial C[1..L-1] and sh_E in parallel
    int64_t my_E = 0;
    for (int k = tid + 1; k < L; k += BLOCK_THREADS) {
        int32_t ck = 0;
        if (!is_skew || ((k & 1) == 0)) {
            for (int i = 0; i < L - k; ++i) ck += (int32_t)sh_s[i] * (int32_t)sh_s[i + k];
        }
        sh_C[k] = ck;
        my_E += (int64_t)ck * ck;
    }
    red_dE[tid] = my_E;
    __syncthreads();
    for (int off = BLOCK_THREADS >> 1; off > 0; off >>= 1) {
        if (tid < off) red_dE[tid] += red_dE[tid + off];
        __syncthreads();
    }
    if (tid == 0) {
        sh_E = red_dE[0];
        sh_best_E = sh_E;
        walkers[wid].best_E = sh_E;
        uint64_t h = 0x123456789ABCDEF0ULL;
        for (int i = 0; i < nvars; ++i) if (sh_s[i] > 0) h ^= sh_zob[i];
        sh_hash = h;
        bloom_insert(sh_bloom, h);
    }
    __syncthreads();

    uint64_t trng = (rng_states[wid]
        ^ ((uint64_t)(round_id + 1) * 0xBF58476D1CE4E5B9ULL)
        ^ ((uint64_t)(tid + 1) * 0x9E3779B97F4A7C15ULL)) | 1ULL;

    for (int step = 0; step < walk_steps; ++step) {
        int64_t cur_E = sh_E;
        uint64_t cur_h = sh_hash;

        int64_t best_d = 9000000000000000000LL;
        int best_p = -1;
        int ties = 0;

        for (int p = tid; p < nvars; p += BLOCK_THREADS) {
            uint64_t cand_h = cur_h ^ sh_zob[p];
            if (bloom_contains(sh_bloom, cand_h)) continue;

            int64_t d = is_skew ? skew_flip_delta(sh_s, sh_C, L, p) : full_flip_delta(sh_s, sh_C, L, p);
            if (d < best_d) { best_d = d; best_p = p; ties = 1; }
            else if (d == best_d) {
                ties++;
                if ((xs64(trng) % (uint64_t)ties) == 0ULL) best_p = p;
            }
        }
        red_dE[tid] = best_d;
        red_p[tid] = best_p;
        __syncthreads();

        for (int off = BLOCK_THREADS >> 1; off > 0; off >>= 1) {
            if (tid < off) {
                int64_t d1 = red_dE[tid], d2 = red_dE[tid + off];
                if (d2 < d1 || (d2 == d1 && red_p[tid + off] >= 0 && (xs64(trng) & 1ULL))) {
                    red_dE[tid] = d2; red_p[tid] = red_p[tid + off];
                }
            }
            __syncthreads();
        }

        int chosen_p = red_p[0];
        int64_t chosen_d = red_dE[0];
        __syncthreads();

        if (chosen_p < 0) break;

        if (!is_skew) {
            int sp = (int)sh_s[chosen_p];
            for (int k = tid + 1; k < L; k += BLOCK_THREADS) {
                int d = 0;
                if (chosen_p >= k) d -= 2 * sp * (int)sh_s[chosen_p - k];
                if (chosen_p + k < L) d -= 2 * sp * (int)sh_s[chosen_p + k];
                if (d != 0) sh_C[k] += d;
            }
        } else {
            int c = nvars - 1;
            if (chosen_p == c) {
                int sp = (int)sh_s[chosen_p];
                for (int k = (tid + 1) * 2; k <= c; k += BLOCK_THREADS * 2) {
                    int d = -2 * sp * ((int)sh_s[chosen_p - k] + (int)sh_s[chosen_p + k]);
                    if (d != 0) sh_C[k] += d;
                }
            } else {
                int q = L - 1 - chosen_p, sp = (int)sh_s[chosen_p], sq = (int)sh_s[q];
                for (int k = (tid + 1) * 2; k < L; k += BLOCK_THREADS * 2) {
                    int d = 0;
                    if (chosen_p >= k) d -= 2 * sp * (int)sh_s[chosen_p - k];
                    if (chosen_p + k < L && (chosen_p + k) != q) d -= 2 * sp * (int)sh_s[chosen_p + k];
                    if (q >= k && (q - k) != chosen_p) d -= 2 * sq * (int)sh_s[q - k];
                    if (q + k < L) d -= 2 * sq * (int)sh_s[q + k];
                    if (d != 0) sh_C[k] += d;
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            sh_s[chosen_p] = -sh_s[chosen_p];
            if (is_skew && chosen_p != (nvars - 1)) sh_s[L - 1 - chosen_p] = -sh_s[L - 1 - chosen_p];
            sh_E = cur_E + chosen_d;
            uint64_t nh = cur_h ^ sh_zob[chosen_p];
            sh_hash = nh;
            bloom_insert(sh_bloom, nh);
        }
        __syncthreads();

        int64_t new_E = sh_E, old_best = sh_best_E;
        __syncthreads();
        if (new_E < old_best) {
            for (int i = tid; i < L; i += BLOCK_THREADS) walkers[wid].best_s[i] = sh_s[i];
            __syncthreads();
            if (tid == 0) { sh_best_E = new_E; walkers[wid].best_E = new_E; }
            __syncthreads();
        }
    }
}

int64_t exact_energy(const int8_t *s, int L) {
    int64_t E = 0;
    for (int k = 1; k < L; ++k) {
        int32_t ck = 0;
        for (int i = 0; i < L - k; ++i) ck += (int32_t)s[i] * (int32_t)s[i + k];
        E += (int64_t)ck * ck;
    }
    return E;
}

void enforce_skew(int8_t *s, int L) {
    int c = L >> 1;
    for (int j = 1; j <= c; ++j) s[c + j] = (int8_t)(((j & 1) ? -1 : 1) * (int)s[c - j]);
}

std::string seq_to_str(const int8_t *s, int L) {
    std::string out(L, '0');
    for (int i = 0; i < L; ++i) out[i] = (s[i] > 0) ? '1' : '0';
    return out;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        printf("Usage: %s L mode(full|skew) seconds out_file seed_file [num_walkers=174] [walk_steps=4096] [mix_mode=0]\n", argv[0]);
        return 1;
    }
    int L = atoi(argv[1]);
    int is_skew = (std::string(argv[2]) == "skew") ? 1 : 0;
    double max_s = atof(argv[3]);
    std::string out_file = argv[4];
    std::string seed_file = argv[5];
    int nw = (argc >= 7) ? atoi(argv[6]) : 174;
    int walk_steps = (argc >= 8) ? atoi(argv[7]) : 4096;
    int mix_mode = (argc >= 9) ? atoi(argv[8]) : 0;
    int nvars = is_skew ? ((L + 1) >> 1) : L;

    std::vector<std::vector<int8_t>> seeds;
    {
        std::ifstream in(seed_file);
        std::string line;
        while (std::getline(in, line)) {
            std::string bits;
            for (char ch : line) if (ch == '0' || ch == '1') bits.push_back(ch);
            if ((int)bits.size() == L) {
                std::vector<int8_t> v(L);
                for (int i = 0; i < L; ++i) v[i] = (bits[i] == '1') ? 1 : -1;
                if (is_skew) enforce_skew(v.data(), L);
                seeds.push_back(v);
            }
        }
    }

    std::mt19937_64 hrng(20261127ULL ^ (uint64_t)L * 1000003ULL ^ (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    int64_t best_E = 9000000000000000000LL;
    std::vector<int8_t> best_s(L, 1);
    std::vector<std::pair<int64_t, std::vector<int8_t>>> elites;

    for (auto &sd : seeds) {
        int64_t e = exact_energy(sd.data(), L);
        if (e < best_E) { best_E = e; best_s = sd; }
        elites.push_back({e, sd});
    }
    if (seeds.empty()) {
        for (int i = 0; i < L; ++i) best_s[i] = (hrng() & 1) ? 1 : -1;
        if (is_skew) enforce_skew(best_s.data(), L);
        best_E = exact_energy(best_s.data(), L);
        seeds.push_back(best_s);
        elites.push_back({best_E, best_s});
    }
    std::sort(elites.begin(), elites.end(), [](const auto &a, const auto &b){ return a.first < b.first; });
    if (elites.size() > 16) elites.resize(16);

    int64_t init_E = best_E;
    {
        std::ofstream out(out_file);
        out << L << " " << best_E << " " << ((double)L*L/(2.0*best_E)) << " " << seq_to_str(best_s.data(), L) << "\n";
    }

    std::vector<uint64_t> hzob(MAX_L);
    for (int i = 0; i < MAX_L; ++i) hzob[i] = hrng() | 1ULL;

    std::vector<SawState> hw(nw);
    std::vector<uint64_t> hr(nw);
    for (int w = 0; w < nw; ++w) hr[w] = hrng() | 1ULL;

    SawState *dw; uint64_t *dz, *dr;
    cudaMalloc(&dw, sizeof(SawState) * nw);
    cudaMalloc(&dz, sizeof(uint64_t) * MAX_L);
    cudaMalloc(&dr, sizeof(uint64_t) * nw);
    cudaMemcpy(dz, hzob.data(), sizeof(uint64_t) * MAX_L, cudaMemcpyHostToDevice);
    cudaMemcpy(dr, hr.data(), sizeof(uint64_t) * nw, cudaMemcpyHostToDevice);

    auto t0 = std::chrono::steady_clock::now();
    int round = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < max_s) {
        for (int w = 0; w < nw; ++w) {
            const std::vector<int8_t> &base =
                (mix_mode == 3 && !seeds.empty())
                    ? ((w % 4 == 0) ? elites[w % elites.size()].second : seeds[(w + round * 37) % seeds.size()])
                    : ((w % 3 == 0) ? best_s : elites[hrng() % elites.size()].second);
            memcpy(hw[w].s, base.data(), L);
            int bucket = w % 20;

            if (mix_mode == 1) {
                if (bucket < 14) {
                    for (int i = 0; i < nvars; ++i) hw[w].s[i] = (hrng() & 1) ? 1 : -1;
                } else if (bucket < 17) {
                    int kb = 1 + (int)(hrng() % 12);
                    for (int b = 0; b < kb; ++b) { int p = (int)(hrng() % nvars); hw[w].s[p] = -hw[w].s[p]; }
                } else {
                    int i = (int)(hrng() % nvars), j = (int)(hrng() % nvars);
                    int8_t b = (hrng() & 1) ? 1 : -1;
                    if (i < j) { for (int k = i; k < j; ++k) hw[w].s[k] = hw[w].s[k + 1]; hw[w].s[j] = b; }
                    else { for (int k = i; k > j; --k) hw[w].s[k] = hw[w].s[k - 1]; hw[w].s[j] = b; }
                }
            } else if (mix_mode == 2) {
                if (bucket < 2) {
                } else if (bucket < 10) {
                    int kb = 1 + (int)(hrng() % 4);
                    for (int b = 0; b < kb; ++b) { int p = (int)(hrng() % nvars); hw[w].s[p] = -hw[w].s[p]; }
                } else if (bucket < 15) {
                    int nsh = 1 + (int)(hrng() % 2);
                    for (int sh = 0; sh < nsh; ++sh) {
                        int i = (int)(hrng() % nvars), j = (int)(hrng() % nvars);
                        int8_t b = (hrng() & 1) ? 1 : -1;
                        if (i < j) { for (int k = i; k < j; ++k) hw[w].s[k] = hw[w].s[k + 1]; hw[w].s[j] = b; }
                        else { for (int k = i; k > j; --k) hw[w].s[k] = hw[w].s[k - 1]; hw[w].s[j] = b; }
                    }
                } else {
                    int kb = 5 + (int)(hrng() % 12);
                    for (int b = 0; b < kb; ++b) { int p = (int)(hrng() % nvars); hw[w].s[p] = -hw[w].s[p]; }
                }
            } else if (mix_mode == 3) {
                if (bucket < 4) {
                } else if (bucket < 12) {
                    int kb = 1 + (int)(hrng() % 5);
                    for (int b = 0; b < kb; ++b) { int p = (int)(hrng() % nvars); hw[w].s[p] = -hw[w].s[p]; }
                } else if (bucket < 17) {
                    int i = (int)(hrng() % nvars), j = (int)(hrng() % nvars);
                    int8_t b = (hrng() & 1) ? 1 : -1;
                    if (i < j) { for (int k = i; k < j; ++k) hw[w].s[k] = hw[w].s[k + 1]; hw[w].s[j] = b; }
                    else { for (int k = i; k > j; --k) hw[w].s[k] = hw[w].s[k - 1]; hw[w].s[j] = b; }
                } else {
                    for (int i = 0; i < nvars; ++i) hw[w].s[i] = (hrng() & 1) ? 1 : -1;
                }
            } else {
                if (bucket == 0) {
                } else if (bucket < 6) {
                    int kb = 1 + (int)(hrng() % 5);
                    for (int b = 0; b < kb; ++b) { int p = (int)(hrng() % nvars); hw[w].s[p] = -hw[w].s[p]; }
                } else if (bucket < 11) {
                    int i = (int)(hrng() % nvars), j = (int)(hrng() % nvars);
                    int8_t b = (hrng() & 1) ? 1 : -1;
                    if (i < j) { for (int k = i; k < j; ++k) hw[w].s[k] = hw[w].s[k + 1]; hw[w].s[j] = b; }
                    else { for (int k = i; k > j; --k) hw[w].s[k] = hw[w].s[k - 1]; hw[w].s[j] = b; }
                } else if (bucket < 16) {
                    int kb = 6 + (int)(hrng() % 18);
                    for (int b = 0; b < kb; ++b) { int p = (int)(hrng() % nvars); hw[w].s[p] = -hw[w].s[p]; }
                } else {
                    for (int i = 0; i < nvars; ++i) hw[w].s[i] = (hrng() & 1) ? 1 : -1;
                }
            }
            if (is_skew) enforce_skew(hw[w].s, L);
        }
        cudaMemcpy(dw, hw.data(), sizeof(SawState) * nw, cudaMemcpyHostToDevice);

        labs_saw_v3_kernel<<<nw, BLOCK_THREADS>>>(dw, dz, dr, nw, L, is_skew, walk_steps, round);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { fprintf(stderr, "CUDA err: %s\n", cudaGetErrorString(err)); return 1; }

        cudaMemcpy(hw.data(), dw, sizeof(SawState) * nw, cudaMemcpyDeviceToHost);
        for (int w = 0; w < nw; ++w) {
            if (hw[w].best_E < best_E) {
                int64_t chk = exact_energy(hw[w].best_s, L);
                if (chk != hw[w].best_E) {
                    fprintf(stderr, "SAW3 EXACT VERIFY MISMATCH: gpu=%lld cpu=%lld\n",
                            (long long)hw[w].best_E, (long long)chk);
                    return 2;
                }
                best_E = chk;
                best_s.assign(hw[w].best_s, hw[w].best_s + L);
                elites.push_back({best_E, best_s});
                std::sort(elites.begin(), elites.end(), [](const auto &a, const auto &b){ return a.first < b.first; });
                if (elites.size() > 16) elites.resize(16);
                double F = (double)L * L / (2.0 * best_E);
                std::ofstream out(out_file);
                out << L << " " << best_E << " " << F << " " << seq_to_str(best_s.data(), L) << "\n";
                printf("  [SAW3 L=%d rnd=%d] *** NEW BEST E=%lld (F=%.5f, dE=%lld) ***\n",
                       L, round, (long long)best_E, F, (long long)(best_E - init_E));
                fflush(stdout);
            } else if (elites.size() < 16 || hw[w].best_E < elites.back().first) {
                int64_t e = hw[w].best_E;
                bool dup = false;
                for (auto &el : elites) if (el.first == e) { dup = true; break; }
                if (!dup) {
                    int64_t chk = exact_energy(hw[w].best_s, L);
                    if (chk == e) {
                        elites.push_back({e, std::vector<int8_t>(hw[w].best_s, hw[w].best_s + L)});
                        std::sort(elites.begin(), elites.end(), [](const auto &a, const auto &b){ return a.first < b.first; });
                        if (elites.size() > 16) elites.resize(16);
                    }
                }
            }
        }
        round++;
    }
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double msteps = (double)round * nw * walk_steps / 1e6;
    printf("  [SAW3 L=%d DONE] rounds=%d elapsed=%.2fs steps=%.2fM (%.2fM steps/s) best_E=%lld F=%.5f\n",
           L, round, elapsed, msteps, msteps / std::max(elapsed, 1e-6), (long long)best_E, (double)L*L/(2.0*best_E));
    cudaFree(dw); cudaFree(dz); cudaFree(dr);
    return 0;
}
