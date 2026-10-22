"""CG_Loader: Orca-style coarse-grained Micro-C target for Evo2HiC-cg.

Evo2HiC-cg is the Seq2HiC model trained on the same target as Orca (Zhou, Nat. Genet. 2022):
the ICE-balanced Micro-C map coarse-grained with cooltools' adaptive_coarsegrain. Only the
target changes; the model input (the expected matrix of the matching .hic file) and the
read-count conditioning are identical to the original Seq2HiC model.

Serves the diagonal bands written by scripts/make_cg_targets.py through the HiC_Loader interface:
  .read_count      from the HiC_Loader of <target_specified>.hic at 2 kb, as in the original Seq2HiC run
  .get(...)        (1, H, W) coarse-grained target; NaN where the target has no value (masked in the loss)
  .get_expected()  the 2 kb expected matrix of the same .hic, summed to the band resolution
Use --augment-resolution equal to the band resolution (4 kb). Only square intra-chromosomal
windows within the band half-width are supported.
"""
import json
import os

import numpy as np

from config import hic_data_dir, cg_target_dir
from dataset.hic_loader import HiC_Loader
from hic_utils import find_chr_in_set


class CG_Loader:
    def __init__(self, target_specified, resolution, read_count='bin-coverage', base_resolution=2000, **kwargs):
        target_dir = os.path.join(cg_target_dir, target_specified)
        with open(os.path.join(target_dir, 'meta.json')) as f:
            self.meta = json.load(f)
        assert resolution == self.meta['res'], f'cg band is at {self.meta["res"]} bp, got {resolution}'
        self.resolution = resolution
        self.half = int(self.meta['half'])
        self.D = resolution // base_resolution
        self.hic = HiC_Loader(os.path.join(hic_data_dir, target_specified + '.hic'),
                              resolution=base_resolution, read_count=read_count)
        self.read_count = self.hic.read_count
        self.chrs = list(self.meta['chroms'].keys())
        self.bands = {c: np.load(os.path.join(target_dir, f'{c}.npy'), mmap_mode='r') for c in self.chrs}

    def get(self, chr0, s0, e0, strand0, chr1, s1, e1, strand1, norm=False):
        c0 = find_chr_in_set(chr0, self.chrs)
        assert chr0 == chr1 and s0 == s1 and e0 == e1, 'CG_Loader: square intra-chromosomal windows only'
        band = self.bands[c0]
        n = band.shape[0]
        b0, b1 = s0 // self.resolution, int(np.ceil(e0 / self.resolution))
        H = b1 - b0
        assert H <= self.half, 'window exceeds band half-width'
        M = np.full((H, H), np.nan, np.float32)
        for i in range(H):
            gi = b0 + i
            if gi < 0 or gi >= n:
                continue
            jlo, jhi = max(0, i - self.half), min(H, i + self.half + 1)
            jlo = max(jlo, -b0)                      # clip to chromosome start
            jhi = min(jhi, n - b0)                   # clip to chromosome end
            if jhi <= jlo:
                continue
            M[i, jlo:jhi] = band[gi, jlo - i + self.half: jhi - i + self.half]
        if strand0:
            M = np.flip(M, -2)
        if strand1:
            M = np.flip(M, -1)
        return M[None].copy()

    def get_expected(self, *position):
        e = self.hic.get_expected(*position)              # (1, h, w) at the base resolution
        d0, d1 = e.shape[-1] // self.D, e.shape[-2] // self.D
        return np.sum(np.reshape(e, (-1, d1, self.D, d0, self.D)), axis=(-1, -3))
