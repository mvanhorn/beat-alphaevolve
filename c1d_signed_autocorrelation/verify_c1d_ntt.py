#!/usr/bin/env python3
"""
Verifier 2 (Independent Multi-Prime NTT + Garner CRT) for C_1d (Signed Third Autocorrelation).
Zero third-party dependencies - Python 3 standard library only.
Shares no code with verify_c1d_kronecker.py.

Deterministically generates k >= 3 distinct 60-bit NTT-friendly primes p = m * 2^L + 1
with verified primitive roots g (product > 2 * N * max|v|^2), computes the exact signed
autoconvolution modulo each prime, reconstructs every exact signed integer coefficient
via Garner's mixed-radix Chinese Remainder Theorem, and checks the deciding knot by
direct O(N) integer summation.
"""
import gzip
import hashlib
import json
import math
import sys
import time
from fractions import Fraction


def is_prime_mr(n):
    """Deterministic Miller-Rabin primality test for all 64-bit integers."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while (d & 1) == 0:
        d >>= 1
        s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_ntt_primes(count, min_pow2, target_bits=60):
    """Deterministically generate `count` distinct NTT primes p = k * 2^min_pow2 + 1 with primitive roots."""
    step = 1 << min_pow2
    k = (1 << (target_bits - min_pow2)) + 1
    found = []
    while len(found) < count:
        p = k * step + 1
        if is_prime_mr(p):
            rem = k
            factors = {2}
            f = 3
            while f * f <= rem:
                while rem % f == 0:
                    factors.add(f)
                    rem //= f
                f += 2
            if rem > 1:
                factors.add(rem)
            g = 2
            while True:
                if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
                    break
                g += 1
            found.append((p, g))
        k += 1
    return found


def ntt_inplace(a, mod, root, invert):
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        wlen = pow(root, (mod - 1) // length, mod)
        if invert:
            wlen = pow(wlen, mod - 2, mod)
        half = length >> 1
        tw = [1] * half
        for k in range(1, half):
            tw[k] = (tw[k - 1] * wlen) % mod
        for i in range(0, n, length):
            for k in range(half):
                u = a[i + k]
                v = (a[i + k + half] * tw[k]) % mod
                s = u + v
                if s >= mod:
                    s -= mod
                d = u - v
                if d < 0:
                    d += mod
                a[i + k] = s
                a[i + k + half] = d
        length <<= 1

    if invert:
        inv_n = pow(n, mod - 2, mod)
        for i in range(n):
            a[i] = (a[i] * inv_n) % mod


def ntt_autoconv_crt(v):
    N = len(v)
    M_len = 2 * N - 1
    L = 1 << M_len.bit_length()
    min_pow2 = L.bit_length() - 1
    max_abs_v = max(abs(x) for x in v)
    bound = 2 * N * max_abs_v * max_abs_v + 1
    need_bits = bound.bit_length() + 10
    n_primes = max(3, (need_bits + 59) // 60)
    primes = find_ntt_primes(n_primes, max(min_pow2, 22), target_bits=60)

    residues = []
    for p, g in primes:
        arr = [x % p for x in v] + [0] * (L - N)
        ntt_inplace(arr, p, g, invert=False)
        for i in range(L):
            arr[i] = (arr[i] * arr[i]) % p
        ntt_inplace(arr, p, g, invert=True)
        residues.append(arr[:M_len])

    mods = [p for p, _ in primes]
    prod = 1
    for p in mods:
        prod *= p
    half_prod = prod >> 1

    inv_table = [[0] * n_primes for _ in range(n_primes)]
    for i in range(n_primes):
        for j in range(i + 1, n_primes):
            inv_table[i][j] = pow(mods[i], mods[j] - 2, mods[j])

    c = [0] * M_len
    for idx in range(M_len):
        x = [0] * n_primes
        for i in range(n_primes):
            val = residues[i][idx]
            for j in range(i):
                val = ((val - x[j]) * inv_table[j][i]) % mods[i]
            x[i] = val
        acc = 0
        mult = 1
        for i in range(n_primes):
            acc += x[i] * mult
            mult *= mods[i]
        if acc > half_prod:
            acc -= prod
        c[idx] = acc
    return c, mods


def verify_file(path):
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

    c, mods = ntt_autoconv_crt(v)

    max_abs = -1
    max_idx = -1
    for m, val in enumerate(c):
        av = abs(val)
        if av > max_abs:
            max_abs = av
            max_idx = m

    lo = 0 if max_idx < N else max_idx - N + 1
    hi = max_idx if max_idx < N else N - 1
    direct = sum(v[i] * v[max_idx - i] for i in range(lo, hi + 1))
    if direct != c[max_idx]:
        raise AssertionError(f"Direct deciding-knot check failed at m={max_idx}")

    num = 2 * N * max_abs
    den = S * S
    g = math.gcd(num, den)
    r_num, r_den = num // g, den // g
    dec_scale = 10 ** 30
    sc = (r_num * dec_scale) // r_den
    dec_str = f"{sc // dec_scale}.{sc % dec_scale:030d}"

    print(f"[NTT-CRT] file={path} ({len(mods)} x 60-bit primes)")
    print(f"  SHA-256  : {sha}")
    print(f"  N        : {N}")
    print(f"  max|v|   : {max(abs(x) for x in v)}")
    print(f"  S        : {S}")
    print(f"  peak knot: m={max_idx} (c[m]={c[max_idx]}, direct check PASS)")
    print(f"  exact    : {r_num} / {r_den}")
    print(f"  decimal  : {dec_str}")
    if "score_num" in data and "score_den" in data:
        if int(data["score_num"]) != r_num or int(data["score_den"]) != r_den:
            raise AssertionError("Stored fraction does not match recomputed exact fraction!")
    if "claim" in data:
        if Fraction(r_num, r_den) > Fraction(str(data["claim"])):
            raise AssertionError(f"Exact bound {dec_str} exceeds claimed bound {data['claim']}!")
    print(f"  elapsed  : {time.time()-t0:.2f}s | RESULT: PASS")
    return r_num, r_den


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 verify_c1d_ntt.py <certificate.json[.gz]>")
        sys.exit(2)
    verify_file(sys.argv[1])
