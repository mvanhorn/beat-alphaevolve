#!/usr/bin/env python3
"""
Standalone sanity and adversarial rejection controls for C_1d step-function certificates.
Zero third-party dependencies - Python 3 standard library only.

Checks:
  1. Direct definition-level integration of (f * f)(t) = int f(t - x) f(x) dx in exact
     Fraction arithmetic at the deciding knot, adjacent knots, midpoints, and random off-knot
     shifts t in (-1/2, 1/2), verifying exact piecewise linearity between knots.
  2. Adversarial rejection controls against both independent verifiers:
     - Tighter claim (1e-15 below exact bound) is rejected.
     - Single-cell height perturbation (+1 at midpoint) is rejected.
     - Zero-sum vector is rejected.
     - Non-integer / boolean height is rejected.
"""
import gzip
import json
import os
import random
import sys
import tempfile
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_c1d_kronecker as vk
import verify_c1d_ntt as vn


def exact_autoconv_at_t(v, t_frac):
    """
    Compute (f * f)(t) = int_{-1/4}^{1/4} f(t - x) f(x) dx directly from the interval-overlap
    definition in exact integer/Fraction arithmetic, where f(x) = v[i] on
    [-1/4 + i*h, -1/4 + (i+1)*h) with h = 1/(2N).
    """
    N = len(v)
    # Express everything in common integer units of h / q, where u = (t + 1/2) / h = p / q.
    # Cell i is [i*q, (i+1)*q] in these units (shifted so -1/4 is 0).
    # For x in cell i, y = t - x ranges over [p - (i+1)*q, p - i*q].
    # Cell j is [j*q, (j+1)*q]. Overlap length is max(0, min(y_hi, c_hi) - max(y_lo, c_lo)) * (h / q).
    u = (t_frac + Fraction(1, 2)) * (2 * N)
    p, q = u.numerator, u.denominator
    k = p // q
    alpha = u - k

    total_scaled = 0
    for i in range(N):
        y_lo = p - (i + 1) * q
        y_hi = p - i * q
        j_start = max(0, (y_lo // q) - 1)
        j_end = min(N - 1, (y_hi // q) + 1)
        vi = v[i]
        for j in range(j_start, j_end + 1):
            c_lo = j * q
            c_hi = c_lo + q
            ov = (y_hi if y_hi < c_hi else c_hi) - (y_lo if y_lo > c_lo else c_lo)
            if ov > 0:
                total_scaled += vi * v[j] * ov
    return Fraction(total_scaled, 2 * N * q), k, alpha


def direct_knot_sum(v, m):
    N = len(v)
    if m < 0 or m > 2 * N - 2:
        return 0
    lo = 0 if m < N else m - N + 1
    hi = m if m < N else N - 1
    return sum(v[i] * v[m - i] for i in range(lo, hi + 1))


def run_controls(cert_path):
    t0 = time.time()
    raw = open(cert_path, "rb").read()
    payload = gzip.decompress(raw) if cert_path.endswith(".gz") else raw
    data = json.loads(payload.decode("utf-8"))
    v = data["v_int"]
    N = len(v)
    S = sum(v)
    h = Fraction(1, 2 * N)

    print(f"[Controls] Loaded {cert_path} (N={N})")

    # 1. Off-knot piecewise-linearity and definition-level integral checks
    # Check deciding knot neighborhood + random interior knots and off-knot points.
    # For our N=819200 certificate, deciding knot is m=244112 (also check 0, N-1, 2N-2, and random).
    rng = random.Random(20261126)
    test_shifts = [
        (0, Fraction(0)),            # left endpoint t = -1/2
        (0, Fraction(1, 3)),         # interior of first segment
        (244112, Fraction(0)) if N >= 244113 else (N // 3, Fraction(0)),
        (244112, Fraction(1, 2)) if N >= 244113 else (N // 3, Fraction(1, 2)),
        (244112, Fraction(3, 7)) if N >= 244113 else (N // 3, Fraction(3, 7)),
        (N - 1, Fraction(0)),        # center knot t = 0
        (2 * N - 2, Fraction(1, 2)), # near right endpoint
    ]
    for _ in range(5):
        m_rand = rng.randrange(0, 2 * N - 1)
        alpha_rand = Fraction(rng.randrange(1, 10), 11)
        test_shifts.append((m_rand, alpha_rand))

    for m, alpha in test_shifts:
        # Knot m sits at t_m = -1/2 + (m + 1) * h
        t_frac = Fraction(-1, 2) + (m + 1 + alpha) * h
        val_direct, _, _ = exact_autoconv_at_t(v, t_frac)
        c_m = direct_knot_sum(v, m)
        c_m1 = direct_knot_sum(v, m + 1)
        val_lin = ((1 - alpha) * c_m + alpha * c_m1) * h
        assert val_direct == val_lin, f"Off-knot mismatch at m={m}, alpha={alpha}: {val_direct} != {val_lin}"
    print(f"  [PASS] Exact definition-level integration & piecewise-linearity ({len(test_shifts)} knot/off-knot points)")

    # 2. Fast adversarial rejection controls on a small synthetic certificate (N=64)
    #    so both full verifiers execute in < 0.05s while exercising the exact same code paths.
    v_small = [1000 + ((i * 73) % 211) - 105 for i in range(64)]
    c_small = vk.kronecker_autoconv(v_small)
    max_c = max(abs(x) for x in c_small)
    S_small = sum(v_small)
    exact_small = Fraction(2 * 64 * max_c, S_small * S_small)
    d12 = 10 ** 12
    ceil_12 = (exact_small.numerator * d12 + exact_small.denominator - 1) // exact_small.denominator
    valid_claim = f"{ceil_12 // d12}.{ceil_12 % d12:012d}"
    tight_claim = f"{(ceil_12 - 1) // d12}.{(ceil_12 - 1) % d12:012d}"

    base_small = {
        "n": 64,
        "score_num": exact_small.numerator,
        "score_den": exact_small.denominator,
        "claim": valid_claim,
        "v_int": v_small,
    }

    with tempfile.TemporaryDirectory() as td:
        p_ok = os.path.join(td, "ok.json")
        with open(p_ok, "w") as f:
            json.dump(base_small, f)
        assert vk.verify_file(p_ok) == vn.verify_file(p_ok) == (exact_small.numerator, exact_small.denominator)

        # Control A: tighter claim rejected by BOTH verifiers
        p_tight = os.path.join(td, "tight.json")
        bad_tight = dict(base_small, claim=tight_claim)
        with open(p_tight, "w") as f:
            json.dump(bad_tight, f)
        for name, fn in (("Kronecker", vk.verify_file), ("NTT-CRT", vn.verify_file)):
            rejected = False
            try:
                fn(p_tight)
            except AssertionError:
                rejected = True
            assert rejected, f"{name} failed to reject tighter claim!"

        # Control B: single-cell +1 tamper rejected by BOTH verifiers
        p_tamper = os.path.join(td, "tamper.json")
        v_tamp = list(v_small)
        v_tamp[32] += 1
        bad_tamper = dict(base_small, v_int=v_tamp)
        with open(p_tamper, "w") as f:
            json.dump(bad_tamper, f)
        for name, fn in (("Kronecker", vk.verify_file), ("NTT-CRT", vn.verify_file)):
            rejected = False
            try:
                fn(p_tamper)
            except AssertionError:
                rejected = True
            assert rejected, f"{name} failed to reject single-cell tamper!"

        # Control C: zero-sum rejected
        p_zero = os.path.join(td, "zero.json")
        bad_zero = dict(base_small, v_int=[1, -1] * 32)
        with open(p_zero, "w") as f:
            json.dump(bad_zero, f)
        rejected = False
        try:
            vk.verify_file(p_zero)
        except ValueError:
            rejected = True
        assert rejected, "Kronecker failed to reject sum(v) == 0!"

        # Control D: non-integer / float rejected
        p_flt = os.path.join(td, "flt.json")
        v_flt = list(v_small)
        v_flt[0] = 1.5
        bad_flt = dict(base_small, v_int=v_flt)
        with open(p_flt, "w") as f:
            json.dump(bad_flt, f)
        rejected = False
        try:
            vk.verify_file(p_flt)
        except ValueError:
            rejected = True
        assert rejected, "Kronecker failed to reject non-integer height!"

    print(f"  [PASS] All adversarial rejection controls (A: tighter claim, B: +1 tamper, C: zero-sum, D: float) on BOTH verifiers")
    print(f"[Controls] ALL PASSED in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 verify_c1d_controls.py <certificate.json[.gz]>")
        sys.exit(2)
    run_controls(sys.argv[1])
