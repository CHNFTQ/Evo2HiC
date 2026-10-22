import numpy as np
import hicstraw
from scipy.ndimage import binary_dilation

from hic_utils import *
import utils
from utils import *
from config import *

class HiC_Loader:
    def __init__(
            self,
            
            hic_file,

            resolution,

            read_count = 'bin-coverage',

            chromosome_filter = ['ALL', 'All', 'Y', 'chrY', 'M', 'chrM'],
            
            chr_pos = None,
            
            **kwargs
        ) -> None:

        self.hic_file = hic_file
        self.hic = hicstraw.HiCFile(self.hic_file)
        self.chromosomes = self.hic.getChromosomes()

        if resolution not in self.hic.getResolutions():
            raise ValueError(f'{hic_file} has no {resolution} bp resolution (available: {self.hic.getResolutions()})')

        base_coverage = extract_property(self.hic_file)

        if read_count == 'bin-coverage': 
            self.read_count = base_coverage * resolution
        elif read_count == 'relative-to-human':
            self.read_count = base_coverage * human_total_length
        else:
            raise NotImplementedError

        self.resolution = resolution

        self.reference = self.hic.getGenomeID()

        if len(self.chromosomes) > 2:
            self.store_type = 'original'
            self.chromosome_filter = chromosome_filter

            self.norm_type = 'SCALE'

            self.chrs = []
            for ch0 in self.chromosomes:
                if ch0.name in self.chromosome_filter: continue
                self.chrs.append(ch0.name)
            self.mzds_norm = {}
            self.mzds_raw = {}

            self.norm_vectors = {}

            expected = np.zeros(1000000)
            cnt = np.zeros(1000000)

            for ch0 in self.chromosomes:
                if ch0.name in self.chromosome_filter: continue
                try:
                    self.mzds_norm[ch0.name] = self.hic.getMatrixZoomData(ch0.name, ch0.name, "observed", self.norm_type, "BP", self.resolution)
                except Exception:
                    # some 4DN files only carry KR
                    print(f'HiC_Loader: {self.norm_type} not available in {self.hic_file}, falling back to KR')
                    self.norm_type = 'KR'
                    self.mzds_norm[ch0.name] = self.hic.getMatrixZoomData(ch0.name, ch0.name, "observed", self.norm_type, "BP", self.resolution)
                self.mzds_raw[ch0.name]  = self.hic.getMatrixZoomData(ch0.name, ch0.name, "observed", 'NONE',         "BP", self.resolution)

                # When the file has no such norm vector, hicstraw may return an empty array, an array of
                # the wrong size, or raise. Treat that as "all bins valid" rather than masking every tile to NaN.
                n_bins = int(np.ceil(ch0.length / self.resolution)) + 2
                try:
                    norm_vector = np.asarray(self.mzds_norm[ch0.name].getNormVector(ch0.index), dtype=float)
                except Exception:
                    norm_vector = np.array([])
                if norm_vector.size == 0 or norm_vector.size > 2 * n_bins or not np.isfinite(norm_vector).any():
                    print(f'HiC_Loader: no {self.norm_type} norm vector for {ch0.name} at {self.resolution} bp in {self.hic_file}; all bins treated as valid')
                    norm_vector = np.ones(n_bins, dtype=float)
                elif norm_vector.size < n_bins:
                    norm_vector = np.pad(norm_vector, (0, n_bins - norm_vector.size), constant_values=np.nan)
                self.norm_vectors[ch0.name] = norm_vector

                expected_ch = self.hic.getMatrixZoomData(ch0.name, ch0.name, "expected", self.norm_type, "BP", self.resolution).getExpectedValues()
                expected[:len(expected_ch)] += expected_ch
                cnt[:len(expected_ch)] += 1

            expected = expected/np.maximum(cnt, 1)
            self.expected = np.minimum.accumulate(expected)
        else:
            # all Hi-C are stored in one assembly
            self.store_type = 'assembly'
            assert chr_pos is not None

            self.norm_type = 'NONE'
            
            self.chr_pos = chr_pos
            self.chrs = list(chr_pos.keys())

            self.assembly_name = self.chromosomes[-1].name
            self.mzd = self.hic.getMatrixZoomData(self.assembly_name, self.assembly_name, "observed", self.norm_type, "BP", self.resolution)
            self.expected = self.hic.getMatrixZoomData(self.assembly_name, self.assembly_name, "expected", self.norm_type, "BP", self.resolution).getExpectedValues()

    def get(self, chr0, s0, e0, strand0, chr1, s1, e1, strand1, norm=False):
        c0, c1 = find_chr_in_set(chr0, self.chrs), find_chr_in_set(chr1, self.chrs)

        if self.store_type == 'original':
            assert c0 == c1
            matrix = (self.mzds_norm[c0] if norm else self.mzds_raw[c0]).getRecordsAsMatrix(max(s0, 0), e0-1, max(s1, 0), e1-1)
        elif self.store_type == 'assembly':
            lo0, hi0 = self.chr_pos[c0]
            lo1, hi1 = self.chr_pos[c1]
            matrix = self.mzd.getRecordsAsMatrix(max(lo0+s0, lo0), min(lo0+e0, hi0)-1, max(lo1+s1, lo1), min(lo1+e1, hi1)-1)
        else:
            raise NotImplementedError

        # padding to correct size
        H,  W  = int(np.ceil((e0-s0)/self.resolution)), int(np.ceil((e1-s1)/self.resolution))
        l0, l1 = int(np.ceil(max(0, -s0)/self.resolution)), int(np.ceil(max(0, -s1)/self.resolution))
        r0, r1 = H - matrix.shape[0] - l0, W - matrix.shape[1] - l1
        matrix = np.pad(matrix, ((l0, r0), (l1, r1)), constant_values=(np.nan) if norm else 0)

        if norm:
            mask0 = ~np.isfinite(self.norm_vectors[c0][max(s0, 0) // self.resolution : e0 // self.resolution])
            mask0 = np.pad(mask0, (l0, H - l0 - mask0.shape[0]), constant_values=True)

            mask1 = ~np.isfinite(self.norm_vectors[c1][max(s1, 0) // self.resolution : e1 // self.resolution])
            mask1 = np.pad(mask1, (l1, W - l1 - mask1.shape[0]), constant_values=True)

            matrix[mask0, :] = np.nan
            matrix[:, mask1] = np.nan

        if strand0:
            matrix = np.flip(matrix, -2)
        if strand1:
            matrix = np.flip(matrix, -1)

        matrix = np.expand_dims(matrix, axis=0)

        return matrix.copy()

    def get_expected(self, chr0, s0, e0, strand0, chr1, s1, e1, strand1):
        c0, c1 = find_chr_in_set(chr0, self.chrs), find_chr_in_set(chr1, self.chrs)

        rs0, re0, rs1, re1 = s0//self.resolution, e0//self.resolution, s1//self.resolution, e1//self.resolution
        dis = np.abs(np.arange(rs0, re0)[:, None] - np.arange(rs1, re1)[None, :]).astype(int)

        if self.store_type == 'original':
            assert c0 == c1
            matrix = self.expected[dis]
        elif self.store_type == 'assembly':
            matrix = self.expected[dis]
        else:
            raise NotImplementedError

        if strand0:
            matrix = np.flip(matrix, -2)
        if strand1:
            matrix = np.flip(matrix, -1)

        matrix = np.expand_dims(matrix, axis=0)

        return matrix.copy()

class Downsample_HiC_Loader(HiC_Loader):
    def __init__(
            self,

            hic_file,

            downsample,

            species,

            downsample_target = None,

            **kwargs
        ) -> None:

        self.original_hic_file = hic_file
        self.downsample = downsample
        d_hic_file = downsample_hic(hic_file, downsample, species, downsample_target = downsample_target)

        super().__init__(d_hic_file, **kwargs)

import pandas as pd
def create_loaders_with_index(split, species, high_coverage_hic = False, return_name = False, downsample = None, **kwargs):
    if high_coverage_hic:
        index = pd.read_csv(hic_hc_index, sep='\t')
    else:
        index = pd.read_csv(hic_index, sep='\t')
    index = index[(index['split'] == split) & (index['Organism'] == species)]
    loaders = []
    for _, row in tqdm(index.iterrows(), desc='creating hic loaders', disable=not utils.VERBOSE):
        hic_file = os.path.join(hic_data_dir, row['Hi-C Accession'] + '.hic')
        if not os.path.exists(hic_file):
            print(row['Hi-C Accession'])
            continue
        loader = HiC_Loader(hic_file, **kwargs) if downsample is None else Downsample_HiC_Loader(hic_file, downsample, species, **kwargs)
        if return_name:
            loaders.append((row['Hi-C Accession'], loader))
        else:
            loaders.append(loader)
    return loaders

