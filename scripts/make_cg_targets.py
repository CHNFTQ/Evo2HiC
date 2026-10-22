"""Precompute the Orca-style coarse-grained target used to train Evo2HiC-cg.

For each chromosome of a Micro-C .mcool at 4 kb: ICE-balanced matrix + raw counts ->
cooltools adaptive_coarsegrain (cutoff=5, max_levels=12, Orca's defaults), computed on
~32 Mb blocks (Orca's window size) stepped along the chromosome, keeping only rows at least
~8 Mb from a block edge, and stored as a diagonal band of half-width 512 bins (2 Mb):
    band[i, j] = M[i, i + j - HALF]   (NaN outside the chromosome / where the target is undefined)

adaptive_coarsegrain depends on the 2^k pooling grid, so every block start is anchored to a
multiple of the coarsest pooling scale (1024 bins); the whole chromosome then shares one grid.

Values are multiplied by --scale, one global factor mapping the balanced scale onto the count
scale of the matching .hic file, so the Evo2HiC normaliser works unchanged. We used the median
per-diagonal ratio between the two: 4880 for H1-hESC (mcool 4DNFI9GMP2J8, .hic 4DNFI2TK7L2F)
and 7300 for HFF (mcool 4DNFI643OYP9, .hic 4DNFIPC7P27B).

Output: <cg_target_dir>/<name>/chr<N>.npy and meta.json, where <name> is the accession of the
matching .hic file (passed as --target-specified when training).

Usage: python scripts/make_cg_targets.py <file.mcool> <name> --scale <factor>
"""
import argparse
import json
import os

import cooler
import numpy as np
from cooltools.lib.numutils import adaptive_coarsegrain

from config import cg_target_dir

RES, HALF = 4000, 512
BLOCK, OVER = 8192, 2048         # bins: ~32 Mb blocks, ~8 Mb margin discarded each side
GRID = 1024                      # coarsest adaptive_coarsegrain pooling scale
CHROMS = [f'chr{i}' for i in range(1, 23)] + ['chrX']


def cg_chrom(clr, chrom, n):
    band = np.full((n, 2 * HALF + 1), np.nan, np.float32)
    covered = np.zeros(n, bool)
    pos = 0
    while pos < n:
        last = (n <= BLOCK) or (pos - OVER >= n - BLOCK - GRID)   # tail: one anchored block up to the end
        lo = 0 if last and n <= BLOCK else max(0, ((max(0, n - BLOCK) if last else pos - OVER) // GRID) * GRID)
        hi_req = n if last else lo + BLOCK
        end_bp = min(hi_req * RES, int(clr.chromsizes[chrom]))
        bal = clr.matrix(balance=True).fetch((chrom, lo * RES, end_bp))
        raw = clr.matrix(balance=False).fetch((chrom, lo * RES, end_bp))
        hi = lo + bal.shape[0]
        cg = adaptive_coarsegrain(bal, raw, cutoff=5, max_levels=12).astype(np.float32)
        keep_lo = pos
        keep_hi = n if (last or hi >= n) else hi - OVER
        assert keep_hi > keep_lo, f'{chrom}: no progress at row {pos} (lo={lo}, hi={hi})'
        for i in range(keep_lo, keep_hi):
            jlo, jhi = max(lo, i - HALF), min(hi, i + HALF + 1)
            band[i, jlo - i + HALF: jhi - i + HALF] = cg[i - lo, jlo - lo: jhi - lo]
        covered[keep_lo:keep_hi] = True
        pos = keep_hi
    assert covered.all(), f'{chrom}: {(~covered).sum()} rows never computed'
    return band


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mcool', help='Micro-C .mcool file containing a 4 kb resolution')
    parser.add_argument('name', help='output name: the accession of the matching .hic file')
    parser.add_argument('--scale', type=float, required=True, help='balanced -> .hic count scale factor')
    args = parser.parse_args()

    out = os.path.join(cg_target_dir, args.name)
    os.makedirs(out, exist_ok=True)
    clr = cooler.Cooler(f'{args.mcool}::/resolutions/{RES}')
    meta = dict(mcool=os.path.basename(args.mcool), res=RES, half=HALF, scale=args.scale, cutoff=5, max_levels=12, chroms={})
    for chrom in CHROMS:
        if chrom not in clr.chromnames:
            continue
        n = int(np.ceil(clr.chromsizes[chrom] / RES))
        band = cg_chrom(clr, chrom, n) * args.scale
        np.save(os.path.join(out, f'{chrom}.npy'), band)
        meta['chroms'][chrom] = n
        print(f'[{args.name}] {chrom} n={n} finite={np.isfinite(band).mean():.3f}', flush=True)
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)


if __name__ == '__main__':
    main()
