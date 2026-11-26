# Improved Upper Bound for the Signed Third Autocorrelation Constant C_1d

Permanent Zenodo DOI (preprint PDF, LaTeX source, exact certificate, and independent verifiers): https://doi.org/10.5281/zenodo.24573271

Exact rational step-function certificate establishing

    C_1d <= 1949516888592680059852669907037867980800 / 1343285009105095348743847474054775337361
         =  1.451305475292588934708300581686...
         <= 1.451305475292589

where C_1d is the signed third autocorrelation constant tracked as constant `1d` in Terence Tao's `teorth/optimizationproblems` (Problem 6.4 of Georgiev, Gomez-Serrano, Tao, and Wagner, `arXiv:2511.02864`):

    max_{t in R} | (f * f)(t) | >= C_1d * ( int_{-1/4}^{1/4} f(x) dx )^2

over real-valued (signed) functions `f` supported on `[-1/4, 1/4]`.

This improves the prior best upper bound `[Can2026b]` (`1.451306191849896925...`, printed as `1.4513061918499*`, N = 102,400 cells) by `7.16557308e-7` and is the first certified upper bound below `1.451306`.

## Certificate Summary

- Grid: `N = 819,200` equal cells on `[-1/4, 1/4]`, cell width `h = 1 / 1,638,400`.
- Integer cell heights: `v_0, ..., v_{819199}` in `Z` with `max_i |v_i| = 72057594037927928` (approximately `2^56`) and `S = sum_i v_i = 293206822196766257848`.
- Exact piecewise-linearity: `f * f` is continuous and piecewise linear on `[-1/2, 1/2]` with knots at `t_m = -1/2 + (m + 1)h` (`m = 0, ..., 2N - 2`). Because `|.|` is convex, `max_{t in R} |(f * f)(t)|` is attained at a knot, where `(f * f)(t_m) = h * c_m` with `c_m = sum_{i+j=m} v_i v_j`. Hence

      max_t |(f * f)(t)| / (int f)^2 = (2 * N * max_m |c_m|) / S^2.

- Deciding knot: unique maximum attained at `m = 244112` with `c[244112] = 76153003460651564837994918243666718`.
- Peak flatness: `735,626` of the `1,638,399` knots lie within `1e-10` of the maximum.
- Exact reduced fraction: `1949516888592680059852669907037867980800 / 1343285009105095348743847474054775337361`.
- Exact 30-digit decimal: `1.451305475292588934708300581686...` (rounded up at 15 decimal places to `1.451305475292589`).

## Independent Exact Verification (Zero Dependencies)

Two independent Python 3 standard-library verifiers sharing no code both compute the identical reduced fraction under isolated mode (`python3 -I`):

```bash
# Verifier 1: Kronecker big-integer substitution + mod (2^61 - 1) polynomial check + direct O(N) deciding-knot sum
python3 -I verify_c1d_kronecker.py c1d_upper_n819200.json.gz

# Verifier 2: 3 x 60-bit prime Number-Theoretic Transform + Garner mixed-radix CRT + direct O(N) deciding-knot sum
python3 -I verify_c1d_ntt.py c1d_upper_n819200.json.gz

# Sanity & adversarial controls: exact definition-level overlap integration at 12 knot/off-knot points
# + rejection of tighter claims, +1 single-cell tampering, zero-sum vectors, and non-integer heights
python3 -I verify_c1d_controls.py c1d_upper_n819200.json.gz
```

Both verifiers were also calibrated on the prior record `[Can2026b]` (`c1d_upper_n102400_20261121.json`, SHA-256 `3c668a4f5158499b498c84bf955894d05100f01f709e3c5dde149631766ecc9f`) and independently reproduced its exact fraction `5136552493465657446805004533831259136 / 3539261750767002528079406787843004249` at `m = 28802`.

## Optimization Method & Resolution Ladder

Starting from the published integer heights of `[Can2026b]` (`N = 102,400`, data only; no external optimizer or checker code was read or run), we ran pure-CUDA `fp64` L-BFGS (`m = 50`) on a two-sided LogSumExp smoothing of `max_m |c_m|` with a smoothed inverse-spectrum Fourier preconditioner `P(k) = (|F(k)|^2 + alpha^2)^{-p/2}` and heat-and-quench continuation (`beta` from `2e5` up to `1e12`), followed by unpreconditioned spatial polishing at high `beta`. Cells were split 2x at each refinement step (`N = 102,400 -> 204,800 -> 409,600 -> 819,200`) on a single NVIDIA A100-SXM4-40GB GPU (~7.3 Colab compute units):

| Resolution `N` | Upper bound (`C_1d`) | Gain vs `[Can2026b]` | Flat knots (within `1e-10`) |
| --- | --- | --- | --- |
| `102,400` (`[Can2026b]` seed) | `1.451306191849897` | `0` | - |
| `102,400` (ours) | `1.451306186952053` | `-4.90e-9` | `97,153` |
| `204,800` (ours) | `1.451305922344204` | `-2.695e-7` | `192,582` |
| `409,600` (ours) | `1.451305695176898` | `-4.967e-7` | `378,416` |
| `819,200` (ours, certified) | `1.451305475292589` | `-7.166e-7` | `735,626` |

Final `fp64` heights were scaled to `max_i |v_i| = 2^56` and rounded to exact integers before certification.

## Files and SHA-256 Checksums

| File | SHA-256 |
| --- | --- |
| `c1d_upper_n819200.json.gz` | `d6cad1be8a0d898be77a51ea0e27267770ab8f34f7db8f88702349dbaf4dd7a4` |
| `verify_c1d_kronecker.py` | `320e5642a845920771ea0e97fdd987305a045005d6db6102f96a08bbaa143240` |
| `verify_c1d_ntt.py` | `8c799fafebb3132dca9d5441efa3f1d6f685ae31e1e30940e4ddeeef4c004f0e` |
| `verify_c1d_controls.py` | `d20f5841445877fc791c4e667282ffd540f05b165d91d7c09e13e2049d9a44e3` |
| `gpu_c1d_deep.py` | `13221426d8a0a9032dcad31aea60536f40d84247173a4547119a12256ba5749d` |

## Disclosure

Directed by Matt Van Horn (GitHub: `mvanhorn`), who takes responsibility for the submission. The GPU optimizer, exact verifiers, controls, and writeup were prepared with AI assistance (Google Antigravity / Gemini). Only the published numerical values of `[Can2026b]` were used as a warm start; no other contributor's code or checker was read or executed. No independent third party has replayed this certificate yet.
