"""Auxiliary 1-D targets for Evo2HiC-cg: Orca's cell-matched peak sets binned at 4 kb
(scripts/make_aux_peaks.py -> <cg_target_dir>/<name>/peaks_chr*.npy, uint8 (n_feat, n_bins)).
get() returns a float32 (n_feat, L) window at the dataset resolution, reversed along the genome
axis for strand 1 (the same flip as the DNA input)."""
import json
import os

import numpy as np

from config import cg_target_dir


class PeakLoader:
    def __init__(self, name, resolution=4000):
        root = os.path.join(cg_target_dir, name)
        with open(os.path.join(root, 'peaks_meta.json')) as f:
            meta = json.load(f)
        assert meta['res'] == resolution, f'peaks binned at {meta["res"]}, dataset resolution {resolution}'
        self.res = resolution
        self.features = meta['features']
        self.n_feat = len(self.features)
        self.arr = {}
        for f in os.listdir(root):
            if f.startswith('peaks_chr') and f.endswith('.npy'):
                self.arr[f[len('peaks_'):-4]] = np.load(os.path.join(root, f), mmap_mode='r')

    def _key(self, chrom):
        c = str(chrom)
        return c if c.startswith('chr') else 'chr' + c

    def get(self, chrom, s, e, strand):
        a = self.arr[self._key(chrom)]
        b0, L = s // self.res, (e - s) // self.res
        out = np.zeros((self.n_feat, L), np.float32)
        lo, hi = max(b0, 0), min(b0 + L, a.shape[1])
        if hi > lo:
            out[:, lo - b0: hi - b0] = a[:, lo:hi]
        if strand:
            out = out[:, ::-1]
        return np.ascontiguousarray(out)
