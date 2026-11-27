// Pure CUDA C++ Massively Parallel Memetic Tabu Search Engine for LABS
// (Low-Autocorrelation Binary Sequences / Golay Merit Factor F = L^2 / (2E)).
// Supports BOTH:
//   - "full": unconstrained {-1, +1}^L search (even & odd L, breaks skew-symmetry!)
//   - "skew": exact skew-symmetric search for odd L=2m-1 (s[c+j] = (-1)^j s[c-j])
// Every CUDA block runs an independent Tabu walker in Shared Memory (256 threads
// evaluate ALL L candidate flips in parallel in O(L) time per step!).
// Host performs Memetic Crossover + Diverse Elite Pool Management + Exact O(L^2) verification.
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
#include <sstream>
#include <algorithm>
#include <chrono>
#include <random>

#define MAX_L 1024
#define BLOCK_THREADS 256

struct WalkerState {
    int8_t s[MAX_L];
    int64_t best_E;
    int8_t best_s[MAX_L];
};

__device__ __forceinline__ uint64_t xs64(uint64_t &s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}

// Compute exact delta_E for flipping single bit p in full unconstrained mode
__device__ __forceinline__ int64_t full_flip_delta(const int8_t *s, const int32_t *C, int L, int p) {
    int sp = (int)s[p];
    int64_t dE = 0;
    int kmax1 = (p < L - 1 - p) ? p : (L - 1 - p);
    for (int k = 1; k <= kmax1; ++k) {
        int sum = (int)s[p - k] + (int)s[p + k];
        if (sum != 0) {
            int d = -2 * sp * sum;
            dE += (int64_t)d * (2 * C[k] + d);
        }
    }
    for (int k = kmax1 + 1; k <= p; ++k) {
        int d = -2 * sp * (int)s[p - k];
        dE += (int64_t)d * (2 * C[k] + d);
    }
    for (int k = kmax1 + 1; k < L - p; ++k) {
        int d = -2 * sp * (int)s[p + k];
        dE += (int64_t)d * (2 * C[k] + d);
    }
    return dE;
}

// Compute exact delta_E for flipping free variable p in skew-symmetric mode (L = 2m - 1 odd, p in 0..m-1).
// For p == m-1 (center bit): flips ONLY s[p].
// For p < m-1: simultaneously flips s[p] and s[q] where q = L - 1 - p.
// All odd lags C[2j+1] remain identically 0! Only even lags k = 2, 4, ..., L-1 change!
__device__ __forceinline__ int64_t skew_flip_delta(const int8_t *s, const int32_t *C, int L, int p) {
    int m = (L + 1) >> 1;
    int c = m - 1;
    int64_t dE = 0;
    if (p == c) {
        int sp = (int)s[p];
        for (int k = 2; k <= c; k += 2) {
            int sum = (int)s[p - k] + (int)s[p + k];
            if (sum != 0) {
                int d = -2 * sp * sum;
                dE += (int64_t)d * (2 * C[k] + d);
            }
        }
        return dE;
    }
    int q = L - 1 - p;
    int sp = (int)s[p];
    int sq = (int)s[q];
    int pq_dist = q - p; // even since q = 2c - p => q - p = 2(c - p) is ALWAYS even!
    for (int k = 2; k < L; k += 2) {
        int d = 0;
        if (p >= k) d -= 2 * sp * (int)s[p - k];
        if (p + k < L && (p + k) != q) d -= 2 * sp * (int)s[p + k];
        if (q >= k && (q - k) != p) d -= 2 * sq * (int)s[q - k];
        if (q + k < L) d -= 2 * sq * (int)s[q + k];
        // Note: when k == pq_dist, pair (p, q) flips BOTH signs so s[p]*s[q] is UNCHANGED!
        if (d != 0) {
            dE += (int64_t)d * (2 * C[k] + d);
        }
    }
    return dE;
}

__global__ void labs_tabu_kernel(
    WalkerState *walkers,
    uint64_t *rng_states,
    int num_walkers,
    int L,
    int is_skew,
    int steps_per_launch,
    int min_tenure,
    int max_tenure,
    int perturb_bits) // if >0, apply random perturbation before Tabu run!
{
    int wid = blockIdx.x;
    if (wid >= num_walkers) return;
    int tid = threadIdx.x;

    __shared__ int8_t sh_s[MAX_L];
    __shared__ int32_t sh_C[MAX_L];
    __shared__ int32_t sh_tabu[MAX_L];
    __shared__ int64_t sh_E;
    __shared__ int64_t sh_best_E;
    __shared__ uint64_t sh_rng;
    __shared__ int sh_chosen_p;

    __shared__ int64_t red_dE[BLOCK_THREADS];
    __shared__ int red_p[BLOCK_THREADS];

    int nvars = is_skew ? ((L + 1) >> 1) : L;

    // Load sequence and initialize tabu table
    for (int i = tid; i < L; i += BLOCK_THREADS) {
        sh_s[i] = walkers[wid].s[i];
        sh_tabu[i] = 0;
    }
    if (tid == 0) {
        sh_best_E = walkers[wid].best_E;
        sh_rng = rng_states[wid];
    }
    __syncthreads();

    // Optional perturbation at start of launch to escape local basin!
    if (tid == 0 && perturb_bits > 0) {
        int kb = perturb_bits;
        // Vary perturbation strength per walker: some light (2-4), some medium (5-12), some heavy (13-24)
        int rtype = wid % 4;
        if (rtype == 0) kb = 0; // pure descent from crossover/elite!
        else if (rtype == 1) kb = 2 + (int)(xs64(sh_rng) % 4);
        else if (rtype == 2) kb = 5 + (int)(xs64(sh_rng) % 8);
        else kb = 10 + (int)(xs64(sh_rng) % 16);

        for (int b = 0; b < kb; ++b) {
            int p = (int)(xs64(sh_rng) % nvars);
            sh_s[p] = -sh_s[p];
            if (is_skew && p != (nvars - 1)) {
                sh_s[L - 1 - p] = -sh_s[L - 1 - p];
            }
        }
    }
    __syncthreads();

    // Compute exact initial C[1..L-1] in parallel across 256 threads!
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
    if (tid == 0) sh_E = red_dE[0];
    __syncthreads();

    uint64_t trng = (rng_states[wid] ^ ((uint64_t)(tid + 1) * 0x9e3779b97f4a7c15ULL)) | 1ULL;

    // Diversify tabu tenure slightly per walker
    int w_min_t = min_tenure + (wid % 7);
    int w_max_t = max_tenure + (wid % 17);
    if (w_max_t <= w_min_t) w_max_t = w_min_t + 4;

    // Main Tabu Search Loop
    for (int step = 1; step <= steps_per_launch; ++step) {
        int64_t cur_E = sh_E;
        int64_t cur_best = sh_best_E;

        int64_t best_d = 9000000000000000000LL;
        int best_p = -1;
        int ties = 0;

        for (int p = tid; p < nvars; p += BLOCK_THREADS) {
            int64_t d = is_skew ? skew_flip_delta(sh_s, sh_C, L, p) : full_flip_delta(sh_s, sh_C, L, p);
            bool asp = (cur_E + d < cur_best);
            if (!asp && sh_tabu[p] > step) continue;
            if (d < best_d) {
                best_d = d; best_p = p; ties = 1;
            } else if (d == best_d) {
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
                    red_dE[tid] = d2;
                    red_p[tid] = red_p[tid + off];
                }
            }
            __syncthreads();
        }

        int chosen_p = red_p[0];
        int64_t chosen_d = red_dE[0];
        __syncthreads();

        if (chosen_p < 0) break;

        // Parallel update of C[1..L-1] across all 256 threads!
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
                int q = L - 1 - chosen_p;
                int sp = (int)sh_s[chosen_p];
                int sq = (int)sh_s[q];
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
            if (is_skew && chosen_p != (nvars - 1)) {
                sh_s[L - 1 - chosen_p] = -sh_s[L - 1 - chosen_p];
            }
            sh_E = cur_E + chosen_d;
            sh_tabu[chosen_p] = step + w_min_t + (int)(xs64(sh_rng) % (w_max_t - w_min_t + 1));
        }
        __syncthreads();

        int64_t new_E = sh_E;
        int64_t old_best = sh_best_E;
        __syncthreads();
        if (new_E < old_best) {
            for (int i = tid; i < L; i += BLOCK_THREADS) walkers[wid].best_s[i] = sh_s[i];
            __syncthreads();
            if (tid == 0) {
                sh_best_E = new_E;
                walkers[wid].best_E = new_E;
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < L; i += BLOCK_THREADS) walkers[wid].s[i] = sh_s[i];
    if (tid == 0) rng_states[wid] = sh_rng;
}

// ---------------- Host Exact O(L^2) Verification & Memetic Evolution ----------------

int64_t exact_energy(const int8_t *s, int L) {
    int64_t E = 0;
    for (int k = 1; k < L; ++k) {
        int32_t ck = 0;
        for (int i = 0; i < L - k; ++i) ck += (int32_t)s[i] * (int32_t)s[i + k];
        E += (int64_t)ck * ck;
    }
    return E;
}

bool check_skew_host(const int8_t *s, int L) {
    if ((L & 1) == 0) return false;
    int c = L >> 1;
    for (int j = 1; j <= c; ++j) {
        int exp = ((j & 1) ? -1 : 1) * (int)s[c - j];
        if ((int)s[c + j] != exp) return false;
    }
    return true;
}

void enforce_skew(int8_t *s, int L) {
    int c = L >> 1;
    for (int j = 1; j <= c; ++j) {
        s[c + j] = (int8_t)(((j & 1) ? -1 : 1) * (int)s[c - j]);
    }
}

uint64_t hash_seq(const int8_t *s, int L) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < L; ++i) {
        h ^= (uint64_t)(s[i] > 0 ? 1 : 0);
        h *= 1099511628211ULL;
    }
    return h;
}

std::string seq_to_str(const int8_t *s, int L) {
    std::string out(L, '0');
    for (int i = 0; i < L; ++i) out[i] = (s[i] > 0) ? '1' : '0';
    return out;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        printf("Usage: %s L mode(full|skew) seconds out_file seed_file [num_walkers=464] [steps_per_round=1500]\n", argv[0]);
        return 1;
    }
    int L = atoi(argv[1]);
    std::string mode = argv[2];
    int is_skew = (mode == "skew") ? 1 : 0;
    double max_s = atof(argv[3]);
    std::string out_file = argv[4];
    std::string seed_file = argv[5];
    int nw = (argc >= 7) ? atoi(argv[6]) : 464; // 8 blocks/SM on L4 (58 * 8 = 464)
    int steps_per_round = (argc >= 8) ? atoi(argv[7]) : 1500;

    if (L >= MAX_L || (is_skew && (L % 2 == 0))) {
        fprintf(stderr, "Invalid L=%d mode=%s\n", L, mode.c_str()); return 1;
    }

    // Load 1 or more seed sequences from seed_file (one 0/1 string of length L per line)
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
    if (seeds.empty()) {
        fprintf(stderr, "No valid seeds of length %d in %s\n", L, seed_file.c_str()); return 1;
    }

    std::mt19937_64 hrng(20261126ULL ^ (uint64_t)L * 999983ULL ^ (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());

    int64_t best_E = 9000000000000000000LL;
    std::vector<int8_t> best_s(L);
    std::vector<std::pair<int64_t, std::vector<int8_t>>> pop; // memetic elite pool
    std::vector<uint64_t> pop_hashes;

    for (auto &sd : seeds) {
        int64_t e = exact_energy(sd.data(), L);
        if (e < best_E) { best_E = e; best_s = sd; }
        uint64_t h = hash_seq(sd.data(), L);
        if (std::find(pop_hashes.begin(), pop_hashes.end(), h) == pop_hashes.end()) {
            pop_hashes.push_back(h);
            pop.push_back({e, sd});
        }
    }
    int64_t init_E = best_E;
    double init_F = (double)L * L / (2.0 * init_E);
    printf("[LABS-GPU L=%d %s] seeds=%zu init_E=%lld (F=%.5f) walkers=%d secs=%.1f\n",
           L, mode.c_str(), seeds.size(), (long long)init_E, init_F, nw, max_s);
    fflush(stdout);

    // Always write initial best so out_file exists
    {
        std::ofstream out(out_file);
        out << L << " " << best_E << " " << init_F << " " << seq_to_str(best_s.data(), L) << "\n";
    }

    std::vector<WalkerState> hw(nw);
    std::vector<uint64_t> hr(nw);
    for (int w = 0; w < nw; ++w) {
        const auto &sd = seeds[w % seeds.size()];
        memset(&hw[w], 0, sizeof(WalkerState));
        memcpy(hw[w].s, sd.data(), L);
        memcpy(hw[w].best_s, sd.data(), L);
        hw[w].best_E = exact_energy(sd.data(), L);
        hr[w] = hrng() | 1ULL;
    }

    WalkerState *dw; uint64_t *dr;
    cudaMalloc(&dw, sizeof(WalkerState) * nw);
    cudaMalloc(&dr, sizeof(uint64_t) * nw);
    cudaMemcpy(dw, hw.data(), sizeof(WalkerState) * nw, cudaMemcpyHostToDevice);
    cudaMemcpy(dr, hr.data(), sizeof(uint64_t) * nw, cudaMemcpyHostToDevice);

    int nvars = is_skew ? ((L + 1) >> 1) : L;
    int min_t = std::max(6, nvars / 28);
    int max_t = std::max(18, nvars / 9);

    auto t0 = std::chrono::steady_clock::now();
    int round = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < max_s) {
        int perturb = (round == 0) ? 1 : 1; // rtype==0 walkers get 0 perturb, others get 2..25 bits!
        labs_tabu_kernel<<<nw, BLOCK_THREADS>>>(
            dw, dr, nw, L, is_skew, steps_per_round, min_t, max_t, perturb);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { fprintf(stderr, "CUDA err: %s\n", cudaGetErrorString(err)); return 1; }

        cudaMemcpy(hw.data(), dw, sizeof(WalkerState) * nw, cudaMemcpyDeviceToHost);

        // Harvest elites & verify exact O(L^2) energy on any new record!
        bool improved = false;
        for (int w = 0; w < nw; ++w) {
            if (hw[w].best_E < best_E) {
                int64_t chk = exact_energy(hw[w].best_s, L);
                if (chk != hw[w].best_E) {
                    fprintf(stderr, "CRITICAL BUG: walker %d E mismatch gpu=%lld exact=%lld!\n",
                            w, (long long)hw[w].best_E, (long long)chk);
                    return 2;
                }
                if (is_skew && !check_skew_host(hw[w].best_s, L)) {
                    fprintf(stderr, "CRITICAL BUG: skew violated!\n");
                    return 2;
                }
                best_E = chk;
                best_s.assign(hw[w].best_s, hw[w].best_s + L);
                improved = true;
            }
            // Add distinct good local optima to memetic elite pool
            if (hw[w].best_E <= best_E + (int64_t)(best_E / 15) + 400) {
                uint64_t h = hash_seq(hw[w].best_s, L);
                if (std::find(pop_hashes.begin(), pop_hashes.end(), h) == pop_hashes.end()) {
                    pop_hashes.push_back(h);
                    pop.push_back({hw[w].best_E, std::vector<int8_t>(hw[w].best_s, hw[w].best_s + L)});
                }
            }
        }

        // Trim elite pool to best 64 diverse sequences
        if (pop.size() > 64) {
            std::sort(pop.begin(), pop.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
            pop.resize(64);
            pop_hashes.clear();
            for (const auto &pr : pop) pop_hashes.push_back(hash_seq(pr.second.data(), L));
        }

        if (improved) {
            double F = (double)L * L / (2.0 * best_E);
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::ofstream out(out_file);
            out << L << " " << best_E << " " << F << " " << seq_to_str(best_s.data(), L) << "\n";
            printf("  [L=%d %s t=%.1fs rnd=%d] *** NEW BEST E=%lld (F=%.5f, dE=%lld) *** pool=%zu\n",
                   L, mode.c_str(), el, round, (long long)best_E, F, (long long)(best_E - init_E), pop.size());
            fflush(stdout);
        }

        // Memetic recombination + elite reseeding for next round!
        for (int w = 0; w < nw; ++w) {
            int psize = (int)pop.size();
            if (w < nw / 4) {
                // Keep best elite (or top-8 elite) directly -> kernel applies 0/light/medium/heavy perturb!
                const auto &p1 = pop[hrng() % std::min(psize, 8)].second;
                memcpy(hw[w].s, p1.data(), L);
            } else if (w < (3 * nw) / 4 && psize >= 2) {
                // 2-point crossover between two elites!
                const auto &p1 = pop[hrng() % std::min(psize, 16)].second;
                const auto &p2 = pop[hrng() % psize].second;
                int a = (int)(hrng() % nvars), b = (int)(hrng() % nvars);
                if (a > b) std::swap(a, b);
                memcpy(hw[w].s, p1.data(), L);
                for (int i = a; i <= b; ++i) hw[w].s[i] = p2[i];
                if (is_skew) enforce_skew(hw[w].s, L);
            } else {
                // Random elite + block shift/flip or keep current walker trajectory for deep exploration!
                if ((w & 1) == 0) {
                    const auto &p1 = pop[hrng() % psize].second;
                    memcpy(hw[w].s, p1.data(), L);
                }
            }
            hw[w].best_E = best_E; // aspiration threshold = global best!
            memcpy(hw[w].best_s, best_s.data(), L);
        }
        cudaMemcpy(dw, hw.data(), sizeof(WalkerState) * nw, cudaMemcpyHostToDevice);

        if (round % 10 == 9) {
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            double tot_steps = (double)(round + 1) * steps_per_round * nw;
            printf("  [L=%d %s t=%.1fs rnd=%d] best E=%lld (F=%.5f) pool=%zu (%.1fM tabu-steps, %.2fM/s)\n",
                   L, mode.c_str(), el, round, (long long)best_E, (double)L*L/(2.0*best_E),
                   pop.size(), tot_steps/1e6, tot_steps/(el*1e6));
            fflush(stdout);
        }
        round++;
    }

    double tot_el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("[LABS-GPU L=%d %s] DONE %.1fs: init_E=%lld -> best_E=%lld (F=%.5f, dE=%lld)\n",
           L, mode.c_str(), tot_el, (long long)init_E, (long long)best_E,
           (double)L*L/(2.0*best_E), (long long)(best_E - init_E));
    cudaFree(dw); cudaFree(dr);
    return 0;
}
