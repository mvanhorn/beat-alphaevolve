# Beat-AlphaEvolve: Exact Rational Open-Math Records

Verified exact-rational mathematical constructions discovered with Colab A100 GPU search + exact Sequential Linear Programming (SLP) / Convex-Concave Procedure (CCP) minimax polishing, benchmarked against Google DeepMind's 67-problem AlphaEvolve repository (`arXiv:2511.02864`) and Erich Friedman's Packing Center live November 2026 tables.

Every construction in this repository is snapped to exact rational coordinates / step heights and verified from scratch using pure Python 3 standard-library integer and `fractions.Fraction` arithmetic (no floating-point tolerances, no NumPy, no SciPy).

## Verify everything in 1 second

```bash
python3 verify_all.py
```

## Scoreboard (exact rational verified, Nov 26, 2026)

| Problem | Target | Prior Record | Ours (Exact Q) | Delta | Status |
|---|---|---|---|---|---|
| Low-Autocorrelation Binary Sequences (LABS / Golay merit factor / Bernasconi model) | 270 lengths across `L = 528..1005` (`12` continuation + `258` Legendre-gap/high-`L`) | `labs-continuation` (Oct 2026) & rotated Legendre (`6.342061` asymptotic limit; Dimitrov `arXiv:2206.12070` `6.4194`) | 270 exact-integer records: `L=528` (`F=34848/4937=7.05854`), `L=536` (`6.52115`), `L=566` (`6.45489`), `L=589` (`6.43256`), `L=805` (`6.42907`), `L=970` (`6.42156`); `167` at `L>=587` above `6.342061` | Up to `-1,508` energy (`L=971`) and `+0.15172` merit factor (`L=817`) | 270 new all-time world records (DOI `10.5281/zenodo.24581609`; 2 independent stdlib exact verifiers; see `labs_records_2026/`) |
| AlphaEvolve #4 / Tao `teorth/optimizationproblems` `C_1d` | Signed third autocorrelation upper bound (N=819,200) | `1.4513061918499*` (`[Can2026b]`, Nov 21 2026, N=102,400; AlphaEvolve `1.4688`) | `1.451305475292589` (`1949516888592680059852669907037867980800 / 1343285009105095348743847474054775337361`) | `-7.166e-7` (first under `1.451306`, 3.4x prior step) | New all-time world record (2 independent stdlib exact verifiers; see `c1d_signed_autocorrelation/`) |
| AlphaEvolve #50 / Erich Friedman `maxmin` | 2D max-to-min ratio^2, n=34 | 31.00251 (Yanis Gomes Nov 2026) | `30.904708496562` | `-0.097802` (first under 31) | New all-time world record |
| AlphaEvolve #5: Erdos Minimum Overlap (1955) | Upper bound on C_5 (N=800) | 0.380868 published (SimpleTES Jul 2026; TTT-Discover 0.380876; AlphaEvolve 0.380924); note open Nov 2026 `teorth/optimizationproblems` PR #368 reaches 0.38085647 | `0.380865928860` (`9521648221496422681 / 25000000000000000000`) | `-0.00000207` vs SimpleTES (`-0.0000581` vs AE) | Beats all published papers at N=800 (behind open PR #368) |
| AlphaEvolve #50 (`maxmin`) | 2D max-to-min ratio^2, n=16 | 12.889266112 (AlphaEvolve May 2025) | `12.889229927416` | `-0.000036` vs AE | Beats AlphaEvolve |
| AlphaEvolve #50 (`maxmin3`) | 3D max-to-min ratio^2, n=14 | 4.165849767 (AlphaEvolve May 2025) | `4.165783484368` | `-0.000066` vs AE | Beats AlphaEvolve |
| AlphaEvolve #36 (`cirRsqu`) | Sum of radii in unit square, n=26 | 2.6358627 (AlphaEvolve May 2025; Haowei Lin Jul 2026: 2.63598+) | `2.635983079` | `+0.000120` vs AE | Beats AlphaEvolve (exact non-overlapping Q) |
| AlphaEvolve #36 (`cirRsqu`) | Sum of radii in unit square, n=32 | 2.9379445 (AlphaEvolve May 2025; Berthold Jan 2026: 2.93957+) | `2.939572765` | `+0.001628` vs AE | Beats AlphaEvolve (exact non-overlapping Q) |
| Erich Friedman `maxmin` extension | 2D n=41..50 (beyond current n=40 table) | Unlisted | 10 exact rational packings (incl. exact `r^2 = 39` at n=42) | Frontier | New frontier certificates |
| Erich Friedman `maxmin3` extension | 3D n=31..40 (beyond current n=30 table) | Unlisted | 10 exact rational packings | Frontier | New frontier certificates |

## Why these work (method summary)

1. Erdos Minimum Overlap (#5): AlphaEvolve's published N=95 construction (0.380924) sits in an even-symmetric saddle point. Breaking symmetry via iterated asymmetric basin-hopping + exact active-set trust-region Sequential Linear Programming (HiGHS) across non-multiple grid refinements (`N = 300 -> 500 -> 600 -> 800`) drives the continuous overlap supremum down to `0.380865928860`. Because `f` and `1 - f` are uniform step functions on `[-1, 1]`, their convolution is piecewise linear with knots strictly at grid shifts `k * (2/N)`, so the exact integer discrete maximum equals the exact continuous supremum with zero approximation gap.
2. Max-to-Min Distance Ratio (#50): 6,144-to-16,384 parallel GPU basins on an NVIDIA A100 using annealed LogSumExp smooth minimax (`log(max d^2) - log(min d^2)`) with hexagonal/FCC/concentric-shell initializations, followed by exact trust-region SLP minimax polishing on the top 100 basins, then exact integer numerator snapping over `D = 10^9`.
3. Circle Packing Max Sum of Radii (#36): 4,096 parallel GPU basins followed by exact monotonic Convex-Concave Procedure (CCP) Sequential LP, then exact integer feasibility enforcement `(x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2` (eliminating float overlap artifacts).

## Repository layout

- `labs_records_2026/`: 270 exact-integer verified LABS world records (`L = 528..1005`), two independent zero-dependency Python stdlib verifiers (`verify_labs_v1.py`, `verify_labs_v2.py`), custom CUDA shared-memory Bloom-filter SAW & Memetic Tabu engines (`labs_gpu_saw_v3.cu`, `labs_gpu_destroyer.cu`), 2-panel frontier figure, 4-page technical note PDF, and `SHA256SUMS`
- `c1d_signed_autocorrelation/`: `C_1d` N=819,200 world-record certificate (`c1d_upper_n819200.json.gz`), two independent zero-dependency Python stdlib exact verifiers (`verify_c1d_kronecker.py`, `verify_c1d_ntt.py`), off-knot/adversarial controls (`verify_c1d_controls.py`), CUDA `fp64` optimizer (`gpu_c1d_deep.py`), `SHA256SUMS`, and technical note
- `verify_all.py`: standalone Python 3 stdlib verifier (no dependencies) for the AlphaEvolve #5 / #36 / #50 suite
- `certificates/`: exact rational JSON certificates
- `visuals/`: `mm34.gif` (225x224 Erich Friedman format), `mm34_hires.png`, `erdos5_world_record.png`
- `SUBMISSION_DRAFTS.md`: ready-to-send submission drafts (not submitted yet)

## Total GPU compute

One Colab A100 session for ~55 minutes (~7.5 compute units / ~$0.75) plus local Apple M5 Max CPU polishing.
