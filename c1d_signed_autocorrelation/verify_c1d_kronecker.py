#!/usr/bin/env python3
"""
Verifier 1 (Kronecker Big-Integer Substitution) for C_1d (Signed Third Autocorrelation Constant).
Zero third-party dependencies — Python 3 standard library ONLY.

For a step function f on [-1/4, 1/4] with N equal cells of width h = 1/(2N) and signed integer
heights v_0, ..., v_{N-1}:
  - f * f is continuous and piecewise linear on [-1/2, 1/2], with knots at t_m = -1/2 + (m+1)h
    for m = 0, ..., 2N-2 (and 0 at the endpoints -1/2, 1/2).
  - By convexity of |.| on each linear segment, max_{t in R} |f*f(t)| is attained at a knot:
      (f * f)(t_m) = h * sum_{i+j=m} v_i v_j = h * c_m.
  - Integral: int f = h * S where S = sum_i v_i.
  - Therefore:
      C_1d <= max_t |f*f(t)| / (int f)^2 = (2 * N * max_m |c_m|) / S^2.

This verifier computes ALL 2N-1 exact integer coefficients c_m using Kronecker substitution
(packing shifted nonnegative heights into a single arbitrary-precision integer and squaring),
independently verifies the deciding knot by direct O(N) integer summation, checks the polynomial
identity modulo 2^61 - 1, and checks exact rational inequality against any claimed upper bound.
"""
import gzip
import hashlib
import json
import math
import sys
import time
from fractions import Fraction


def kronecker_autoconv(v):
    """Exact signed integer autoconvolution c[m] = sum_{i+j=m} v[i]*v[j] via Kronecker substitution."""
    N = len(v)
    if N == 0:
        return []
    M_abs = max(abs(x) for x in v)
    if M_abs == 0:
        return [0] * (2 * N - 1)

    # Shift to strictly nonnegative: u_i = v_i + B where B = M_abs
    # Then 0 <= u_i <= 2*B, so sum_{i+j=m} u_i u_j <= N * (2*B)^2 = 4 * N * B^2.
    B = M_abs
    max_conv_u = 4 * N * B * B
    W = max_conv_u.bit_length() + 4  # guaranteed zero carry between slots!
    mask = (1 << W) - 1

    # Pack u_i into a single integer P = sum_{i=0}^{N-1} (v_i + B) * 2^{i * W}
    # Use divide-and-conquer packing in O(N log N) instead of O(N^2) shifts!
    def pack(arr, lo, hi):
        if hi - lo <= 64:
            acc = 0
            for idx in range(hi - 1, lo - 1, -1):
                acc = (acc << W) | (arr[idx] + B)
            return acc
        mid = (lo + hi) // 2
        low_val = pack(arr, lo, mid)
        high_val = pack(arr, mid, hi)
        return low_val | (high_val << ((mid - lo) * W))

    P = pack(v, 0, N)
    P2 = P * P

    # Unpack 2N-1 coefficients via divide-and-conquer in O(N log N)!
    M_len = 2 * N - 1
    u_conv = [0] * M_len

    def unpack(val, lo, hi):
        if hi - lo <= 64:
            cur = val
            for idx in range(lo, hi):
                u_conv[idx] = cur & mask
                cur >>= W
            return
        mid = (lo + hi) // 2
        shift = (mid - lo) * W
        low_part = val & ((1 << shift) - 1)
        high_part = val >> shift
        unpack(low_part, lo, mid)
        unpack(high_part, mid, hi)

    unpack(P2, 0, M_len)

    # Undo shift:
    # sum_{i+j=m} (v_i + B)(v_j + B) = c_m + 2*B * sum_{i+j=m} v_i + k_m * B^2
    # where for a given m, i ranges over [lo_m, hi_m] = [max(0, m-(N-1)), min(N-1, m)],
    # k_m = hi_m - lo_m + 1, and sum_{i=lo_m}^{hi_m} v_i is computed in O(1) via prefix sums!
    pref = [0] * (N + 1)
    for i, x in enumerate(v):
        pref[i + 1] = pref[i] + x

    c = [0] * M_len
    B2 = B * B
    twoB = 2 * B
    for m in range(M_len):
        lo = 0 if m < N else m - N + 1
        hi = m if m < N else N - 1
        cnt = hi - lo + 1
        sv = pref[hi + 1] - pref[lo]
        c[m] = u_conv[m] - twoB * sv - cnt * B2
    return c


def verify_file(path, benchmark_frac=None):
    t0 = time.time()
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    payload = gzip.decompress(raw) if path.endswith(".gz") else raw
    data = json.loads(payload.decode("utf-8"))
    v = data["v_int"]
    if not isinstance(v, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        raise ValueError("v_int must be a list of exact integers")
    N = len(v)
    if "n" in data and data["n"] != N:
        raise ValueError(f"n mismatch: {data['n']} != {N}")
    S = sum(v)
    if S == 0:
        raise ValueError("sum(v) == 0")

    c = kronecker_autoconv(v)

    # Modular polynomial identity check modulo Mersenne prime p = 2^61 - 1
    MOD = (1 << 61) - 1
    for x_eval in (3, 17, 131071):
        pv = 0
        xp = 1
        for val in v:
            pv = (pv + (val % MOD) * xp) % MOD
            xp = (xp * x_eval) % MOD
        pc = 0
        xp = 1
        for val in c:
            pc = (pc + (val % MOD) * xp) % MOD
            xp = (xp * x_eval) % MOD
        if (pv * pv) % MOD != pc:
            raise AssertionError(f"Polynomial identity failed mod 2^61-1 at x={x_eval}")

    # Find deciding knot(s)
    max_abs = -1
    max_idx = -1
    for m, val in enumerate(c):
        av = abs(val)
        if av > max_abs:
            max_abs = av
            max_idx = m

    # Direct O(N) exact integer summation at deciding knot max_idx!
    lo = 0 if max_idx < N else max_idx - N + 1
    hi = max_idx if max_idx < N else N - 1
    direct_sum = sum(v[i] * v[max_idx - i] for i in range(lo, hi + 1))
    if direct_sum != c[max_idx]:
        raise AssertionError(f"Direct sum mismatch at m={max_idx}: {direct_sum} != {c[max_idx]}")

    num = 2 * N * max_abs
    den = S * S
    g = math.gcd(num, den)
    r_num = num // g
    r_den = den // g
    frac = Fraction(r_num, r_den)

    # High-precision decimal string (30 digits) via integer division
    dec_scale = 10 ** 30
    scaled = (r_num * dec_scale) // r_den
    dec_str = f"{scaled // dec_scale}.{scaled % dec_scale:030d}"

    dt = time.time() - t0
    print(f"[Kronecker] file={path}")
    print(f"  SHA-256  : {sha}")
    print(f"  N        : {N}")
    print(f"  max|v|   : {max(abs(x) for x in v)}")
    print(f"  S        : {S}")
    print(f"  peak knot: m={max_idx} (c[m]={c[max_idx]}, direct check PASS)")
    print(f"  exact    : {r_num} / {r_den}")
    print(f"  decimal  : {dec_str}")
    if "score_num" in data and "score_den" in data:
        ok_stored = (int(data["score_num"]) == r_num and int(data["score_den"]) == r_den)
        print(f"  matches stored score_num/score_den: {ok_stored}")
        if not ok_stored:
            raise AssertionError("Stored fraction does not match recomputed exact fraction!")
    if "claim" in data:
        claim_frac = Fraction(str(data["claim"]))
        ok_claim = (frac <= claim_frac)
        print(f"  exact <= claim ({data['claim']}): {ok_claim}")
        if not ok_claim:
            raise AssertionError(f"Exact bound {dec_str} exceeds claimed bound {data['claim']}!")
    if benchmark_frac is not None:
        beats = frac < benchmark_frac
        print(f"  strictly beats benchmark: {beats}")
    print(f"  elapsed  : {dt:.2f}s | RESULT: PASS")
    return r_num, r_den


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 verify_c1d_kronecker.py <certificate.json>")
        sys.exit(2)
    verify_file(sys.argv[1])
