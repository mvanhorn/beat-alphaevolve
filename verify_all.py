#!/usr/bin/env python3
"""Zero-dependency exact rational/integer verifier for all Beat-AlphaEvolve certificates.

Uses ONLY Python 3 standard library (`fractions.Fraction`, `math.isqrt`, `json`, `itertools`).
Zero NumPy, zero SciPy, zero floating-point tolerances. Run with `python3 verify_all.py`.
"""
from __future__ import annotations

import itertools
import json
from fractions import Fraction
from pathlib import Path

if not __debug__:
    raise RuntimeError("Run without -O so assertions are active.")

CERTS = Path(__file__).resolve().parent / "certificates"


def verify_erdos_p5():
    d = json.loads((CERTS / "erdos_minimum_overlap_p5.json").read_text())
    N, D = int(d["N"]), int(d["D"])
    f = [int(v) for v in d["numerators"]]
    assert len(f) == N
    assert all(0 <= v <= D for v in f), "Value out of [0, 1]"
    assert sum(f) * 2 == D * N, "Mean of f is not exactly 1/2"
    # Continuous supremum of (f * (1-f))(x) for uniform step function equals discrete max at grid shifts.
    g_rev = [D - v for v in reversed(f)]
    max_conv = 0
    for s in range(2 * N - 1):
        lo = max(0, s - N + 1)
        hi = min(N - 1, s)
        acc = 0
        for i in range(lo, hi + 1):
            acc += f[i] * g_rev[s - i]
        if acc > max_conv:
            max_conv = acc
    bound = Fraction(2 * max_conv, N * D * D)
    assert bound == Fraction(int(d["exact_upper_bound_numerator"]), int(d["exact_upper_bound_denominator"]))
    assert bound < Fraction(380868, 1000000), "Did not beat SimpleTES 0.380868"
    print(f"PASS [Problem #5 Erdos Minimum Overlap, N={N}]: C_5 <= {float(bound):.12f} (exact {bound.numerator}/{bound.denominator})")


def verify_maxmin(path: Path, expected_beat: Fraction | None = None):
    d = json.loads(path.read_text())
    nums = d["point_numerators"]
    assert len(nums) == d["n"]
    assert all(len(p) == d["d"] for p in nums)
    min_d2 = None
    max_d2 = None
    for p, q in itertools.combinations(nums, 2):
        d2 = sum((a - b) * (a - b) for a, b in zip(p, q))
        assert d2 > 0
        if min_d2 is None or d2 < min_d2:
            min_d2 = d2
        if max_d2 is None or d2 > max_d2:
            max_d2 = d2
    r2 = Fraction(max_d2, min_d2)
    assert r2 == Fraction(int(d["exact_r2_numerator"]), int(d["exact_r2_denominator"]))
    if expected_beat is not None:
        assert r2 < expected_beat, f"{r2} not < {expected_beat}"
    print(f"PASS [{d['problem']}]: r^2 = {float(r2):.12f}")


def verify_maxmin_frontier(path: Path):
    data = json.loads(path.read_text())
    for k, d in sorted(data.items(), key=lambda kv: int(kv[0])):
        nums = d["point_numerators"]
        assert len(nums) == d["n"]
        min_d2, max_d2 = None, None
        for p, q in itertools.combinations(nums, 2):
            d2 = sum((a - b) * (a - b) for a, b in zip(p, q))
            assert d2 > 0
            if min_d2 is None or d2 < min_d2:
                min_d2 = d2
            if max_d2 is None or d2 > max_d2:
                max_d2 = d2
        r2 = Fraction(max_d2, min_d2)
        assert r2 == Fraction(int(d["exact_r2_numerator"]), int(d["exact_r2_denominator"]))
    print(f"PASS [{path.name}]: all {len(data)} exact rational frontier configurations verified")


def verify_cirRsqu(path: Path, ae_bound: Fraction):
    d = json.loads(path.read_text())
    n, D = int(d["n"]), int(d["D"])
    xs, ys, rs = d["x_numerators"], d["y_numerators"], d["r_numerators"]
    assert len(xs) == len(ys) == len(rs) == n
    for i in range(n):
        assert 0 <= rs[i] <= xs[i] <= D - rs[i], f"x boundary violation at {i}"
        assert 0 <= rs[i] <= ys[i] <= D - rs[i], f"y boundary violation at {i}"
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            assert dx * dx + dy * dy >= (rs[i] + rs[j]) ** 2, f"Overlap between {i} and {j}"
    sr = Fraction(sum(rs), D)
    assert sr == Fraction(int(d["exact_sum_r_numerator"]), int(d["exact_sum_r_denominator"]))
    assert sr > ae_bound, f"{sr} not > {ae_bound}"
    print(f"PASS [{d['problem']}]: sum(r) = {float(sr):.12f} (strictly non-overlapping in exact Z)")


if __name__ == "__main__":
    verify_erdos_p5()
    verify_maxmin(CERTS / "maxmin_2d_n34.json", Fraction(3100251, 100000))
    verify_maxmin(CERTS / "maxmin_2d_n16.json", Fraction(12889266112, 1000000000))
    verify_maxmin(CERTS / "maxmin_3d_n14.json", Fraction(4165849767, 1000000000))
    verify_cirRsqu(CERTS / "cirRsqu_n26.json", Fraction(26358627, 10000000))
    verify_cirRsqu(CERTS / "cirRsqu_n32.json", Fraction(29379445, 10000000))
    verify_maxmin_frontier(CERTS / "maxmin_2d_frontier_41_50.json")
    verify_maxmin_frontier(CERTS / "maxmin_3d_frontier_31_40.json")
    print("\nALL EXACT RATIONAL CERTIFICATES PASSED CLEAN-ROOM VERIFICATION.")
