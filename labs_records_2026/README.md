# 339 New Record Low-Autocorrelation Binary Sequences (L = 528 to 1100)

Permanent Zenodo archive and concept DOI: https://doi.org/10.5281/zenodo.24581609 (`10.5281/zenodo.24581609`)

Exact-integer and exact-rational verified world records for the Low-Autocorrelation Binary Sequences (LABS / Golay merit factor / Bernasconi Ising model) problem across lengths `L = 528..1100` (Version 2, expanded from 270 records in Version 1).

## Summary

- Total strictly verified new world records: 339 (zero 4-decimal rounding ties; every sequence has strictly lower exact integer sidelobe energy `E` than all five published 2022-2026 baseline tables).
- Continuation frontier (`L = 528..536` and `582..586`): 12 new records beating the October 2026 `labs-continuation` frontier, led by `L = 528` (`E = 19,748`, `F = 34848/4937 = 7.05854`, `-92` energy), `L = 536` (`E = 22,028`, `F = 6.52115`, `-520` energy, `+0.15039`), and `L = 586` (`E = 26,837`, `F = 6.39781`, `-360` energy, `+0.08469`).
- Mid-range dead zone (`L = 537..566`): 28 of 30 lengths broken (all except `L = 540, 541`) via two-stage frozen skew chirp-head search (`labs_gpu_chirp_ore.cu`) plus bidirectional continuation, opening a new `F = 6.50-6.70` island topped by `L = 563` (`E = 23,665`, `F = 6.69700`, `-1,432` energy, `+0.38212`), `L = 561` (`E = 23,560`, `F = 6.67914`, `-1,412` energy, `+0.37766`), `L = 559` (`F = 6.65221`, `+0.35498`), and `L = 551` (`F = 6.54905`, `+0.27293`).
- High-length regime (`L = 587..1005`): 261 new records, with 175 exceeding the Jedwab-Katz-Schmidt `6.342061...` asymptotic rotated-Legendre limit, 29 exceeding `F = 6.40`, and 6 exceeding `6.4194` (`L = 589, 805, 806, 969, 970, 971`), topped by `L = 589` (`F = 6.43256`) and `L = 971` (`-1,508` energy).
- Extended frontier (`L = 1006..1100`): 38 new records, led by `L = 1006` beating Dimitrov (`arXiv:2206.12070`) at `E = 79,491`, `F = 506018/79491 = 6.36573` (vs `6.3568`, `-408` energy), plus `L = 1049` (`F = 6.34486`), `L = 1097` (`F = 6.34308`), and `L = 1100` (`F = 6.33654`).
- Total compute cost: 19.71 Google Colab compute units (~US$1.97) across two NVIDIA L4 GPUs.

![LABS 2026 World Record Frontier](labs_frontier_2026.png)

## Independent Exact Verification (Zero Dependencies)

Both verifiers use only Python 3 standard library exact integer arithmetic and share zero code:

```bash
/usr/bin/python3 -I verify_labs_v1.py labs_records_2026_verified.tsv
/usr/bin/python3 -I verify_labs_v2.py labs_records_2026_verified.tsv
shasum -a 256 -c SHA256SUMS
```

- `verify_labs_v1.py`: Direct `O(L^2)` integer autocorrelation sum and `fractions.Fraction(L*L, 2*E)`.
- `verify_labs_v2.py`: Independent big-integer XOR-popcount identity `C_k = (L - k) - 2 * popcount((x ^ (x >> k)) & mask)`.

## Files

- `labs_records_2026_verified.tsv`: All 339 new world-record binary sequences (`1 = +1`, `0 = -1`) with exact integer `E`, exact reduced rational `F = L^2 / (2E)`, decimal `F`, prior best `E`/`F`/source, and gains.
- `labs_records_2026_summary.json`: Machine-readable summary metadata.
- `labs_records_2026.pdf` / `labs_records_2026.tex`: Technical report (Version 2).
- `labs_frontier_2026.png` / `labs_frontier_2026.pdf`: Two-panel comparison figure against 2022-2026 baselines (`L = 524..1100`).
- `labs_gpu_saw_v3.cu`, `labs_gpu_saw_v4.cu`: Custom CUDA 131,072-bit shared-memory Bloom-filter Self-Avoiding Walk engines (up to `L = 1280`, zero DRAM traffic in the walk loop).
- `labs_gpu_chirp_ore.cu`: Two-stage frozen skew chirp-head CUDA self-avoiding walk engine that broke the `L = 537..566` dead zone.
- `labs_gpu_destroyer.cu`: Custom CUDA 464-walker shared-memory Memetic Tabu Search engine.
- `verify_labs_v1.py`, `verify_labs_v2.py`: Two independent zero-dependency exact verifiers.
- `SHA256SUMS`: Cryptographic hashes for all files in this directory.
