"""Bin Orca's auxiliary 1-D targets into 4 kb bins for the Evo2HiC-cg auxiliary loss.

Orca ships, per cell type, a sorted BED of peaks (<cell>.hg38.bed.sorted.gz, 4th column = feature
name) and the list of features (<cell>.hg38.bed.sorted.features): Roadmap Epigenomics peak sets,
32 for H1-hESC and 22 for HFF. A bin is 1 for a feature if any peak of that feature overlaps it,
as in Orca's MultibinGenomicFeatures(bin=4000, step=4000, mode='any').

Run scripts/make_cg_targets.py first: the number of bins per chromosome is read from its meta.json.
Output: <cg_target_dir>/<name>/peaks_chr*.npy (uint8, n_feat x n_bins) and peaks_meta.json.

Usage: python scripts/make_aux_peaks.py <peaks.bed.sorted.gz> <peaks.bed.sorted.features> <name>
"""
import gzip
import json
import os
import sys

import numpy as np

from config import cg_target_dir

RES = 4000


def main(bed, features, name):
    out = os.path.join(cg_target_dir, name)
    with open(features) as f:
        feats = [l.strip() for l in f if l.strip()]
    fidx = {f: i for i, f in enumerate(feats)}
    with open(os.path.join(out, 'meta.json')) as f:
        nbins = json.load(f)['chroms']
    arr = {c: np.zeros((len(feats), n), np.uint8) for c, n in nbins.items()}
    n_used = n_skip = 0
    with gzip.open(bed, 'rt') as fh:
        for line in fh:
            c, s, e, f = line.rstrip('\n').split('\t')[:4]
            if c not in arr or f not in fidx:
                n_skip += 1
                continue
            b0, b1 = int(s) // RES, (int(e) - 1) // RES + 1
            arr[c][fidx[f], b0:min(b1, arr[c].shape[1])] = 1
            n_used += 1
    for c, a in arr.items():
        np.save(os.path.join(out, f'peaks_{c}.npy'), a)
    with open(os.path.join(out, 'peaks_meta.json'), 'w') as f:
        json.dump(dict(features=feats, res=RES, n_peaks_used=n_used, n_skipped=n_skip), f, indent=1)
    print(f'[{name}] {len(feats)} features, {n_used} peaks used, {n_skip} skipped')


if __name__ == '__main__':
    main(*sys.argv[1:4])
