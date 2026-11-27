// Pure CUDA C++ 2-Stage Restriction-Class (Frozen Chirp-Head -> Unlocked) GPU-SAW (lssOrel+)
// Stage 1 (stage1_steps): Freezes indices [0 .. frozen_k-1] (the empirical Fresnel chirp head
//   1^r1 0^r2 1^r3 0^r4 ... matching all F > 7.2 LABS records!) while running 100% on-chip
//   Shared-Memory 131,072-bit Bloom-Filter SAW on the free middle bits [frozen_k .. nvars-1].
//   Because the chirp head cannot collapse during initial descent, walkers lock directly into
//   deep F > 6.6-7.3 chirp basins instead of glassy F ~ 5.8 random valleys!
// Stage 2 (stage2_steps): Unlocks ALL [0 .. nvars-1] bits, resets Bloom filter, and runs
//   full-freedom SAW to relax the chirp head & middle to the exact basin bottom!
// Exact host-side O(L^2) integer verification on every improvement.
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

#define MAX_L 1104
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

__global__ void labs_chirp_ore_kernel(
    SawState *walkers,
    const uint64_t *zobrist,
    const uint64_t *rng_states,
    int num_walkers,
    int L,
    int is_skew,
    int frozen_k,
    int stage1_steps,
    int stage2_steps,
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

    int nvars = is_skew ? ((L + 1) >> 1) : L;

    for (int i = tid; i < BLOOM_WORDS; i += BLOCK_THREADS) sh_bloom[i] = 0U;
    for (int i = tid; i < L; i += BLOCK_THREADS) {
        int8_t v = walkers[wid].s[i];
        sh_s[i] = v;
        walkers[wid].best_s[i] = v;
        sh_zob[i] = zobrist[i];
    }
    __syncthreads();

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

    int total_steps = stage1_steps + stage2_steps;
    for (int step = 0; step < total_steps; ++step) {
        // Transition from Stage 1 (frozen prefix) to Stage 2 (fully unlocked from Stage-1 best_s!)
        if (step == stage1_steps && stage1_steps > 0) {
            for (int i = tid; i < BLOOM_WORDS; i += BLOCK_THREADS) sh_bloom[i] = 0U;
            for (int i = tid; i < L; i += BLOCK_THREADS) sh_s[i] = walkers[wid].best_s[i];
            __syncthreads();
            int64_t s2E = 0;
            for (int k = tid + 1; k < L; k += BLOCK_THREADS) {
                int32_t ck = 0;
                if (!is_skew || ((k & 1) == 0)) {
                    for (int i = 0; i < L - k; ++i) ck += (int32_t)sh_s[i] * (int32_t)sh_s[i + k];
                }
                sh_C[k] = ck;
                s2E += (int64_t)ck * ck;
            }
            red_dE[tid] = s2E;
            __syncthreads();
            for (int off = BLOCK_THREADS >> 1; off > 0; off >>= 1) {
                if (tid < off) red_dE[tid] += red_dE[tid + off];
                __syncthreads();
            }
            if (tid == 0) {
                sh_E = red_dE[0];
                uint64_t h = 0x123456789ABCDEF0ULL;
                for (int i = 0; i < nvars; ++i) if (sh_s[i] > 0) h ^= sh_zob[i];
                sh_hash = h;
                bloom_insert(sh_bloom, h);
            }
            __syncthreads();
        }

        int p_start = (step < stage1_steps) ? frozen_k : 0;
        int64_t cur_E = sh_E;
        uint64_t cur_h = sh_hash;

        int64_t best_d = 9000000000000000000LL;
        int best_p = -1;
        int ties = 0;

        for (int p = p_start + tid; p < nvars; p += BLOCK_THREADS) {
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

        if (chosen_p < 0) {
            if (step < stage1_steps) { step = stage1_steps - 1; continue; }
            else break;
        }

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
    if (argc < 7) {
        printf("Usage: %s L mode(full|skew) seconds out_file prefix_file frozen_k [s1_steps=1800] [s2_steps=3200]\n", argv[0]);
        return 1;
    }
    int L = atoi(argv[1]);
    int is_skew = (std::string(argv[2]) == "skew") ? 1 : 0;
    double max_s = atof(argv[3]);
    std::string out_file = argv[4];
    std::string prefix_file = argv[5];
    int frozen_k = atoi(argv[6]);
    int s1_steps = (argc >= 8) ? atoi(argv[7]) : 1800;
    int s2_steps = (argc >= 9) ? atoi(argv[8]) : 3200;
    int nvars = is_skew ? ((L + 1) >> 1) : L;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int nw = prop.multiProcessorCount * 3; // 174 on L4

    // Load prefix/template sequences (can be exact length L; first frozen_k bits used as chirp head, rest randomized!)
    std::vector<std::vector<int8_t>> prefixes;
    {
        std::ifstream in(prefix_file);
        std::string line;
        while (std::getline(in, line)) {
            std::string bits;
            for (char ch : line) if (ch == '0' || ch == '1') bits.push_back(ch);
            if ((int)bits.size() >= frozen_k) {
                std::vector<int8_t> v(bits.size());
                for (size_t i = 0; i < bits.size(); ++i) v[i] = (bits[i] == '1') ? 1 : -1;
                prefixes.push_back(v);
            }
        }
    }

    std::mt19937_64 hrng(2026112777ULL ^ (uint64_t)L * 1000003ULL ^ (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    int64_t best_E = 9000000000000000000LL;
    std::vector<int8_t> best_s(L, 1);
    std::vector<std::pair<int64_t, std::vector<int8_t>>> elites;

    // Seed initial best_s from any full-length prefix
    for (auto &pf : prefixes) {
        if ((int)pf.size() == L) {
            auto v = pf;
            if (is_skew) enforce_skew(v.data(), L);
            int64_t e = exact_energy(v.data(), L);
            if (e < best_E) { best_E = e; best_s = v; }
            elites.push_back({e, v});
        }
    }
    if (elites.empty()) {
        for (int i = 0; i < nvars; ++i) best_s[i] = (hrng() & 1ULL) ? 1 : -1;
        if (is_skew) enforce_skew(best_s.data(), L);
        best_E = exact_energy(best_s.data(), L);
        elites.push_back({best_E, best_s});
    }
    std::sort(elites.begin(), elites.end(), [](const auto &a, const auto &b){ return a.first < b.first; });
    if (elites.size() > 24) elites.resize(24);
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
            if (!prefixes.empty() && (w % 4 != 0)) {
                // 75%: Take real empirical chirp head [0..frozen_k-1] + random middle [frozen_k..nvars-1]!
                const auto &pf = prefixes[(w + round * 19) % prefixes.size()];
                for (int i = 0; i < frozen_k; ++i) hw[w].s[i] = pf[i];
                // Small 0-2 bit jitter at boundary of chirp head on half of walkers for diversity
                if (w & 1) {
                    int jb = (int)(hrng() % 2);
                    for (int b = 0; b < jb; ++b) {
                        int p = frozen_k - 1 - (int)(hrng() % 15);
                        if (p >= 20) hw[w].s[p] = -hw[w].s[p];
                    }
                }
                for (int i = frozen_k; i < nvars; ++i) hw[w].s[i] = (hrng() & 1ULL) ? 1 : -1;
            } else {
                // 25%: Crossover / perturb best elite middle!
                const auto &el = elites[hrng() % elites.size()].second;
                memcpy(hw[w].s, el.data(), L);
                int kb = 3 + (int)(hrng() % 24);
                for (int b = 0; b < kb; ++b) {
                    int p = frozen_k + (int)(hrng() % std::max(1, nvars - frozen_k));
                    if (p < nvars) hw[w].s[p] = -hw[w].s[p];
                }
            }
            if (is_skew) enforce_skew(hw[w].s, L);
        }
        cudaMemcpy(dw, hw.data(), sizeof(SawState) * nw, cudaMemcpyHostToDevice);

        labs_chirp_ore_kernel<<<nw, BLOCK_THREADS>>>(
            dw, dz, dr, nw, L, is_skew, frozen_k, s1_steps, s2_steps, round);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { fprintf(stderr, "CUDA err: %s\n", cudaGetErrorString(err)); return 1; }

        cudaMemcpy(hw.data(), dw, sizeof(SawState) * nw, cudaMemcpyDeviceToHost);
        for (int w = 0; w < nw; ++w) {
            if (hw[w].best_E < best_E) {
                int64_t chk = exact_energy(hw[w].best_s, L);
                if (chk != hw[w].best_E) { fprintf(stderr, "CHIRP_ORE VERIFY BUG!\n"); return 2; }
                best_E = chk;
                best_s.assign(hw[w].best_s, hw[w].best_s + L);
                elites.push_back({best_E, best_s});
                std::sort(elites.begin(), elites.end(), [](const auto &a, const auto &b){ return a.first < b.first; });
                if (elites.size() > 24) elites.resize(24);
                double F = (double)L * L / (2.0 * best_E);
                std::ofstream out(out_file);
                out << L << " " << best_E << " " << F << " " << seq_to_str(best_s.data(), L) << "\n";
                printf("  [ORE L=%d rnd=%d] *** NEW BEST E=%lld (F=%.5f, dE=%lld) ***\n",
                       L, round, (long long)best_E, F, (long long)(best_E - init_E));
                fflush(stdout);
            } else if (elites.size() < 24 || hw[w].best_E < elites.back().first) {
                int64_t e = hw[w].best_E;
                bool dup = false;
                for (auto &el : elites) if (el.first == e) { dup = true; break; }
                if (!dup && exact_energy(hw[w].best_s, L) == e) {
                    elites.push_back({e, std::vector<int8_t>(hw[w].best_s, hw[w].best_s + L)});
                    std::sort(elites.begin(), elites.end(), [](const auto &a, const auto &b){ return a.first < b.first; });
                    if (elites.size() > 24) elites.resize(24);
                }
            }
        }
        round++;
    }
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("  [ORE L=%d DONE] rounds=%d elapsed=%.1fs best_E=%lld F=%.5f\n",
           L, round, elapsed, (long long)best_E, (double)L*L/(2.0*best_E));
    cudaFree(dw); cudaFree(dz); cudaFree(dr);
    return 0;
}
