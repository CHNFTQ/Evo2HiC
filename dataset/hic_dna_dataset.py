import torch
import numpy as np
from torch.utils.data import default_collate
from torch.utils.data import Dataset, Subset
from beartype.typing import List, Union, Optional
from cooltools.lib.numutils import adaptive_coarsegrain

from dataset.DNA_loader import DNA_Loader
from dataset.hic_loader import HiC_Loader, create_loaders_with_index
from dataset.cool_loader import Cool_Loader
from dataset.mappability_loader import Mappability_Loader
from dataset.evo2_embedding_loader import evo2_Embedding_Loader
from dataset.normalizer import Normalizer
from utils import *
from hic_utils import *
from config import *

def get_with_drop(loader, drop, chr, s, e, strand):
    if drop or loader.option == 'dummy':
        v = loader.get_dummy(chr, s, e, strand)
    else:
        v = loader.get(chr, s, e, strand)

    # add 1 dimension for hic
    return np.expand_dims(v, axis=0).astype(np.float32)

class HiC_DNA_Dataset(Dataset):
    def __init__(
            self,

            resolution,
            chromosome_split,

            HiC_loaders : list[Union[HiC_Loader, Cool_Loader]],
            DNA_loader : DNA_Loader,
            Mappability_loader : Mappability_Loader,

            input_option,
            target_option,
            avg_hic,

            max_separation,
            chunk,
            stride,

            whole_row,

            downsample = 16,
            normalizer : Normalizer = None,

            augment_resolution = None,

            sample = False,
            dataset_size = None,

            flip_prob=0,
            transpose_prob=0,
            DNA_drop_prob=0,
            map_drop_prob=0,
            DNA_shift=0,

            target_downsample_prob=0,
            target_downsample_read_count=0,

            Embedding_loader : evo2_Embedding_Loader = None,

            hic_per_pos = -1,
            pos_per_row = -1,

            map_thres = 0.5,

            training = False,

            area = None,
            seed = 0,
            **kwargs
        ) -> None:
        super().__init__()

        self.HiC_loaders = HiC_loaders
        self.normalizer = normalizer
        self.DNA_loader = DNA_loader
        self.Mappability_loader = Mappability_loader
        self.peak_loader = None      # optional auxiliary 1-D targets (dataset/peak_loader.py), set by the trainer

        self.chromosome_split = chromosome_split

        self.resolution = resolution

        self.input_option = input_option
        self.downsample = downsample
        self.target_option = target_option
        self.avg_hic = avg_hic

        self.max_separation = max_separation
        self.chunk = chunk
        self.stride = stride

        self.augment_resolution = augment_resolution

        self.sample = sample

        self.whole_row = whole_row

        self.flip_prob = flip_prob
        self.transpose_prob = transpose_prob
        self.DNA_drop_prob = DNA_drop_prob
        self.map_drop_prob = map_drop_prob
        self.DNA_shift = DNA_shift

        self.target_downsample_prob = target_downsample_prob
        self.target_downsample_read_count = target_downsample_read_count        

        self.hic_per_pos = hic_per_pos
        self.pos_per_row = pos_per_row

        self.map_thres = map_thres

        self.training = training
        self.seed = seed if training else 0

        self.Embedding_loader = Embedding_loader

        self.area = area

        if self.whole_row:
            assert self.flip_prob == 0
            assert self.transpose_prob == 0
            assert self.DNA_drop_prob == 0

        if self.sample and self.training:
            self.len_p = dataset_size
            self.len_h = len(self.HiC_loaders)
        else:
            self.prepare_position()

            self.len_p = len(self.positions)
            self.len_h = len(self.HiC_loaders)

        print_info(f'Dataset of {self.len_p} genome positions and {self.len_h} HiC datas.')

    def prepare_position(self):
        self.positions = []
        for chid in self.chromosome_split:
            size = self.DNA_loader.get_size(chid)

            stride = self.stride * self.resolution
            chunk = self.chunk * self.resolution

            assert chunk>=stride

            L, R = 0, size
            if self.area is not None:
                L, R = self.area
                assert R-L >= chunk

            R = (R//self.resolution*self.resolution)

            for i in range(L, R, stride):
                if i-stride+chunk >= R: break
                if i+chunk > R: i = R - chunk
                ii = i+ chunk

                if not self.whole_row:
                    for j in range(i, R, stride):
                        if j+chunk > R: j = R - chunk
                        ii, jj = i+ chunk, j + chunk
                        self.positions.append([(chid, i, ii, 0, chid, j, jj, 0)])
                        if j-i+(chunk-stride) >= self.max_separation: break
                else:
                    row = []
                    for j in range(i, size, stride):
                        jj = j + chunk

                        row.append((chid, i, ii, 0, chid, j, jj, 0))
                        if j-i+(chunk-stride) >= self.max_separation: break

                    assert len(row) > 0

                    wanted_len = self.pos_per_row if self.pos_per_row > 0 else np.ceil((self.max_separation - (chunk-stride)) / stride).astype(int) + 1
                    if len(row) < wanted_len:
                        if self.training:
                            continue
                        else:
                            row = row + [row[-1]] * (wanted_len - len(row))

                    assert len(row) >= wanted_len

                    self.positions.append(row)

    def read_hic(self, loaders : List[Union[HiC_Loader, Cool_Loader]], position, seed = 0):
        target_matrices = []
        target_read_counts = []
        input_matrices = []
        input_read_counts = []
        for loader in loaders:
            target_read_count = loader.read_count
            target_submatrix = loader.get(*position, norm=(self.target_option == 'norm'))

            if self.target_downsample_prob > 0 and target_read_count > self.target_downsample_read_count:
                raise NotImplementedError

            if self.input_option == 'LC':
                assert self.target_option == 'hic' or self.target_option == 'norm' or self.target_option == 'hic+norm'
                input_read_count = target_read_count / self.downsample
                if self.training:
                    input_submatrix = np.random.binomial(np.nan_to_num(target_submatrix).astype(int), 1/self.downsample)
                else:
                    rng = np.random.default_rng(seed = seed)
                    input_submatrix = rng.binomial(np.nan_to_num(target_submatrix).astype(int), 1/self.downsample)
            elif self.input_option == 'HC':
                assert self.target_option == 'hic' or self.target_option == 'norm' or self.target_option == 'hic+norm'
                input_read_count = target_read_count
                input_submatrix = np.nan_to_num(target_submatrix)
            elif self.input_option == 'expected':
                input_read_count = target_read_count
                input_submatrix = loader.get_expected(*position)
            else:
                raise NotImplementedError

            if self.target_option == 'hic+norm':
                target_norm_submatrix = loader.get(*position, norm=True)
                target_submatrix = np.concatenate([target_submatrix, target_norm_submatrix], axis=0)

            if self.augment_resolution is not None:
                D = self.resolution // self.augment_resolution
                d0, d1 = target_submatrix.shape[-1]//D, target_submatrix.shape[-2]//D
                target_submatrix = np.sum(np.reshape(target_submatrix, (-1, d0, D, d1, D)), axis=(-1, -3))

                D = self.resolution // self.augment_resolution
                d0, d1 = input_submatrix.shape[-1]//D, input_submatrix.shape[-2]//D
                input_submatrix = np.sum(np.reshape(input_submatrix, (-1, d0, D, d1, D)), axis=(-1, -3))

            target_matrices.append(target_submatrix)
            target_read_counts.append(target_read_count)

            input_matrices.append(input_submatrix)
            input_read_counts.append(input_read_count)

        target_matrix = np.stack(target_matrices, axis=0)
        target_read_count = np.stack(target_read_counts, axis=0)

        input_matrix = np.stack(input_matrices, axis=0)
        input_read_count = np.stack(input_read_counts, axis=0)

        if self.avg_hic:
            assert self.target_option == 'hic' or self.target_option == 'norm' or self.target_option == 'hic+norm'

            target_matrix = np.mean(target_matrix, axis=0, keepdims = True)
            target_read_count = np.mean(target_read_count, axis=0, keepdims = True)

            input_matrix = np.mean(input_matrix, axis=0, keepdims = True)
            input_read_count = np.mean(input_read_count, axis=0, keepdims = True)

        # target_matrix = np.nan_to_num(target_matrix, nan=0)
        # input_matrix = np.nan_to_num(input_matrix, nan=0)
        target_matrix = self.normalizer.normalize(target_matrix).clip(max=1)
        input_matrix = self.normalizer.normalize(input_matrix).clip(max=1)

        return target_matrix.astype(np.float32), np.log(target_read_count).astype(np.float32), input_matrix.astype(np.float32), np.log(input_read_count).astype(np.float32)

    def __getitem__(self, index):
        data = {
            'positions' : [],

            'DNA_col': [],
            'DNA_row': [],
            'mappability_col': [],
            'mappability_row': [],
            'embedding_col' : [],
            'embedding_row' : [],

            'target_matrix': [],
            'target_read_count': [],

            'input_matrix': [],
            'input_read_count': []
        }
        if self.peak_loader is not None:
            data['target_1d'] = []

        if self.sample and self.training:
            ch, s, e, strand = 0, 0, 0, 0
            Find = False
            while not Find:
                ch = np.random.choice(self.chromosome_split)
                size = self.DNA_loader.get_size(ch)
                i = np.random.choice(size//self.resolution-self.chunk)
                ii = i + self.chunk

                ch, s, e, strand = ch, i * self.resolution, ii * self.resolution, 0

                if self.augment_resolution is not None:
                    shift = np.random.choice(np.arange(0, self.resolution, self.augment_resolution))
                    s+=shift
                    e+=shift

                if e>=size:
                    continue

                map = self.Mappability_loader.get(ch, s, e, strand).mean()
                if map < self.map_thres:
                    continue

                if not self.whole_row:
                    positions = [(ch, s, e, strand, ch, s, e, strand)]
                    Find = True
                else:
                    chunk = self.chunk*self.resolution
                    stride = self.stride*self.resolution

                    positions = []
                    for j in range(s, size, stride):
                        jj = j + chunk

                        positions.append((ch, s, e, strand, ch, j, jj, strand))
                        if j-i+(chunk-stride) >= self.max_separation: break

                    assert len(positions) > 0

                    wanted_len = self.pos_per_row if self.pos_per_row > 0 else (self.max_separation - (chunk-stride)) // stride + 1
                    if len(positions) < wanted_len:
                        if self.training:
                            continue
                        else:
                            positions = positions + [positions[-1]] * (wanted_len - len(positions))

                    Find = True
        else:
            positions = self.positions[index]

        if self.pos_per_row>0:
            if self.training:
                ids = np.random.choice(len(positions)-1, self.pos_per_row-1, replace=False) + 1
            else:
                rng = np.random.default_rng(seed = self.seed + index)
                ids = rng.choice(len(positions)-1, self.pos_per_row-1, replace=False) + 1

            ids = sorted(ids)
            positions = positions[:1] + [positions[i] for i in ids]

        for position in positions:
            chr0, s0, e0, strand0, chr1, s1, e1, strand1  = position

            if self.training and np.random.rand() < self.transpose_prob:
                chr0, s0, e0, chr1, s1, e1 = chr1, s1, e1, chr0, s0, e0

            if self.training and np.random.rand() < self.flip_prob:
                strand0, strand1 = 1, 1

            data['positions'].append(((chr0, s0, e0, strand0, chr1, s1, e1, strand1),))

            if self.training and self.DNA_shift>0:
                shift0, shift1 = np.random.randint(-self.DNA_shift, self.DNA_shift, 2)
            else:
                shift0, shift1 = 0, 0

            if self.DNA_loader is not None and self.DNA_loader.option != 'No':
                drop = self.training and np.random.rand() < self.DNA_drop_prob
                if not self.whole_row and self.max_separation > 0:
                    data['DNA_col'].append(get_with_drop(self.DNA_loader, drop, chr0, s0+shift0, e0+shift0, strand0))
                data['DNA_row'].append(get_with_drop(self.DNA_loader, drop, chr1, s1+shift1, e1+shift1, strand1))
            if self.peak_loader is not None:
                data['target_1d'].append(self.peak_loader.get(chr1, s1, e1, strand1))

            if self.Mappability_loader is not None and self.Mappability_loader.option != 'No':
                drop = self.training and np.random.rand() < self.map_drop_prob
                if not self.whole_row and self.max_separation > 0:
                    data['mappability_col'].append(get_with_drop(self.Mappability_loader, drop, chr0, s0+shift0, e0+shift0, strand0))
                data['mappability_row'].append(get_with_drop(self.Mappability_loader, drop, chr1, s1+shift1, e1+shift1, strand1))

            if self.Embedding_loader is not None and self.Embedding_loader.option != 'No':
                if not self.whole_row and self.max_separation > 0:
                    data['embedding_col'].append(get_with_drop(self.Embedding_loader, False, chr0, s0, e0, strand0))
                data['embedding_row'].append(get_with_drop(self.Embedding_loader, False, chr1, s1, e1, strand1))

            if self.hic_per_pos == -1 or self.hic_per_pos > self.len_h:
                hic_indices = [i for i in range(self.len_h)]
            else:
                if self.training:
                    hic_indices = np.random.choice(self.len_h, self.hic_per_pos, replace=False)
                else:
                    rng = np.random.default_rng(seed = self.seed + index)
                    hic_indices = rng.choice(self.len_h, self.hic_per_pos, replace=False)

            target_matrix, target_read_count, input_matrix, input_read_count = \
                self.read_hic([self.HiC_loaders[hi] for hi in hic_indices], (chr0, s0, e0, strand0, chr1, s1, e1, strand1), self.seed+index)

            data['target_matrix'].append(target_matrix)
            data['target_read_count'].append(target_read_count)

            data['input_matrix'].append(input_matrix)
            data['input_read_count'].append(input_read_count)

        for k in data.keys():
            if len(data[k]) > 0:
                data[k] = np.stack(data[k], axis=0)
            else:
                data[k] = np.array([])
        return data

    def __len__(self):
        return self.len_p

class HiC_DNA_Inference_Dataset(HiC_DNA_Dataset):
    def __init__(
            self,

            **kwargs
        ) -> None:
        super().__init__(**kwargs)

    def read_hic(self, HiC_loader : Union[HiC_Loader, Cool_Loader], position, seed = 0):
        raise NotImplementedError
        input_read_count = HiC_loader.read_count
        if self.input_option in ['HC', 'LC']:
            assert self.target_option == 'hic'
            input_submatrix = HiC_loader.get(*position)
            assert hasattr(HiC_loader, 'downsample') ^ (self.input_option == 'HC')
        elif self.input_option == 'expected':
            input_submatrix = HiC_loader.get_expected(*position)
            if hasattr(HiC_loader, 'downsample'):
                input_submatrix = input_submatrix * HiC_loader.downsample
        else:
            raise NotImplementedError

        if hasattr(HiC_loader, 'downsample'):
            target_read_count = input_read_count * HiC_loader.downsample
        else:
            target_read_count = input_read_count

        if self.target_option == 'hic':
            if not self.OE:
                input_submatrix = np.nan_to_num(input_submatrix, nan=0)
                input_submatrix = self.normalizer.normalize(input_submatrix).clip(max=1)
            else:
                expected = HiC_loader.get_expected(*position)
                expected = (input_read_count / target_read_count) * expected
                eps = expected.min()
                input_submatrix = np.log((input_submatrix+eps) / (expected+eps))
                input_submatrix = input_submatrix.clip(min=-2, max=2)
                input_submatrix = np.nan_to_num(input_submatrix, nan=-2.001)

        return np.zeros_like(input_submatrix) , np.log(target_read_count), input_submatrix , np.log(input_read_count)

def prepare_datasets(
    resolution,

    target_option,
    target_specified,
    high_coverage_hic,

    read_count,

    DNA_option,
    evo2_option,

    augment_resolution,

    train_pos_per_row,
    valid_pos_per_row,

    train_hic_per_pos,
    valid_hic_per_pos,

    target_downsample_prob,

    species = 'human',

    **kwargs
):
    augment_resolution = augment_resolution if augment_resolution is not None else resolution
    if target_option == 'hic' or target_option == 'norm' or target_option == 'hic+norm':
        if target_specified is None:
            hic_loaders = create_loaders_with_index('train', species, resolution=augment_resolution, high_coverage_hic=high_coverage_hic, read_count=read_count)
        else:
            hic_file = os.path.join(hic_data_dir, target_specified + '.hic')
            hic_loaders = [HiC_Loader(hic_file, resolution=augment_resolution, read_count=read_count)]
    elif target_option == 'cool':
        hic_loaders = [Cool_Loader('H1ESC', resolution=augment_resolution)]
    elif target_option == 'cg':
        # Evo2HiC-cg: Orca-style coarse-grained target (dataset/cg_loader.py)
        from dataset.cg_loader import CG_Loader
        assert target_specified is not None, '--target-option cg needs --target-specified <accession>'
        hic_loaders = [CG_Loader(target_specified, resolution=augment_resolution, read_count=read_count)]
    else:
        raise NotImplementedError

    DNA_loader = DNA_Loader(DNA_map[species], DNA_option)
    mappability_loader = Mappability_Loader(mappability_map[species], DNA_option)
    embedding_loader = evo2_Embedding_Loader(evo2_embedding_map[species], evo2_option)

    train_set = HiC_DNA_Dataset(
        resolution,
        splits[species]['train'],
        hic_loaders,
        DNA_loader,
        mappability_loader,
        target_option = target_option,
        training=True,
        augment_resolution = augment_resolution,
        pos_per_row=train_pos_per_row,
        hic_per_pos=train_hic_per_pos,
        target_downsample_prob = target_downsample_prob,
        Embedding_loader = embedding_loader,
        **kwargs
    )

    valid_set = HiC_DNA_Dataset(
        resolution,
        splits[species]['valid'],
        hic_loaders,
        DNA_loader,
        mappability_loader,
        target_option = target_option,
        training=False,
        augment_resolution = augment_resolution,
        pos_per_row=valid_pos_per_row,
        hic_per_pos=valid_hic_per_pos,
        target_downsample_prob = 0,
        Embedding_loader = embedding_loader,
        **kwargs
    )

    if target_downsample_prob == 0:
        return train_set, valid_set, None

    sparse_valid_set = HiC_DNA_Dataset(
        resolution,
        splits[species]['valid'],
        hic_loaders,
        DNA_loader,
        mappability_loader,
        target_option = target_option,
        training=False,
        pos_per_row=valid_pos_per_row,
        hic_per_pos=valid_hic_per_pos,
        target_downsample_prob = 1,
        Embedding_loader = embedding_loader,
        **kwargs
    )

    return train_set, valid_set, sparse_valid_set