#!/usr/bin/env python3
"""Independent Zero-Dependency Exact Verifier #1 for LABS (Low-Autocorrelation Binary Sequences).
Recomputes exact integer sidelobe energy E = sum_{k=1}^{L-1} (sum_{i=0}^{L-1-k} s_i s_{i+k})^2
and exact rational merit factor F = L^2 / (2E) using Python arbitrary-precision integers & Fractions.
"""
import sys
from fractions import Fraction

def verify_tsv(path):
    ok = 0
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            L = int(parts[0])
            claimed_E = int(parts[1])
            bits = parts[-1]
            if len(bits) != L or any(c not in "01" for c in bits):
                raise AssertionError(f"Line {ln}: invalid bitstring for L={L}")
            s = [1 if c == "1" else -1 for c in bits]
            E = 0
            for k in range(1, L):
                ck = 0
                for i in range(L - k):
                    ck += s[i] * s[i + k]
                E += ck * ck
            if E != claimed_E:
                raise AssertionError(f"Line {ln} L={L}: E mismatch! claimed={claimed_E}, exact={E}")
            F_exact = Fraction(L * L, 2 * E)
            ok += 1
    print(f"VERIFIER #1 PASS EXACT Z/Q: {ok} sequences verified in {path}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        verify_tsv(p)
