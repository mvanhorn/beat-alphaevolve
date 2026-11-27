#!/usr/bin/env python3
"""Independent Zero-Dependency Exact Verifier #2 for LABS using big-integer bit-packing
(popcount XOR autocorrelation identity: C_k = (L - k) - 2 * popcount((x ^ (x >> k)) & mask_k)).
Shares ZERO code or algorithm with verify_labs_v1.py!
"""
import sys

def verify_tsv_bitpacked(path):
    cnt = 0
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            L = int(cols[0])
            claimed_E = int(cols[1])
            seq = cols[-1]
            if len(seq) != L or set(seq) - {"0", "1"}:
                raise ValueError(f"Line {ln}: malformed sequence")
            # Bit i = 1 if seq[i]=='1' else 0
            x = int(seq[::-1], 2)
            E = 0
            for k in range(1, L):
                n_pairs = L - k
                mask = (1 << n_pairs) - 1
                diff = bin((x ^ (x >> k)) & mask).count("1")
                ck = n_pairs - 2 * diff
                E += ck * ck
            if E != claimed_E:
                raise AssertionError(f"Line {ln} L={L}: bitpacked E={E} != claimed {claimed_E}")
            cnt += 1
    print(f"VERIFIER #2 PASS BIT-PACKED EXACT INTEGER: {cnt} sequences verified in {path}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        verify_tsv_bitpacked(p)
