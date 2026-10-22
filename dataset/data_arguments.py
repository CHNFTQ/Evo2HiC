import argparse

from config import *
from utils import *

def add_hic_dataset_arguments(parser = None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # HiC data param
    HiC_data_arguments = parser.add_argument_group('HiC dataset arguments')
    HiC_data_arguments.add_argument('-r', '--resolution', type=int, choices=res_list, default=2000, help='The resolution of HiC matrices')

    HiC_data_arguments.add_argument('-input', '--input-option', type=str, choices=['HC', 'LC', 'expected'], default='LC', help='Use which matrix as input')
    HiC_data_arguments.add_argument('--downsample', type=float, default = 16, help='downsample rate when input LC')
    HiC_data_arguments.add_argument('-target', '--target-option', type=str, choices=['hic', 'norm', 'hic+norm', 'cool', 'cg'], default='hic', help='Use which matrix as target')
    HiC_data_arguments.add_argument('--target-specified', type=str, default=None)
    HiC_data_arguments.add_argument('--high-coverage-hic', action=argparse.BooleanOptionalAction, default=False, help='Use high-coverage hic')
    HiC_data_arguments.add_argument('--avg-hic', action=argparse.BooleanOptionalAction, default=False, help='average over hic')

    HiC_data_arguments.add_argument('--chunk', type=int, default = 160, help='The size of submatrices')
    HiC_data_arguments.add_argument('--stride', type=int, default = 120, help='The stride to slice submatrices along x and y axis')
    HiC_data_arguments.add_argument('--max-separation', type=int, default = 2000000, help='Focusing on submatrices within this thres.')

    HiC_data_arguments.add_argument('--read-count', type=str, default = 'bin-coverage', choices=['bin-coverage', 'relative-to-human'], help='how to measure read count')

    Normalize_arguments = parser.add_argument_group('Normalize arguments')

    Normalize_arguments.add_argument('--normalization', type=str, choices= ['none', 'mixed', 'log1p', 'linear'], default='mixed', help='the method to normalize Hi-C')
    Normalize_arguments.add_argument('--max-reads', type=float, default=5000, help='max number of reads in a pixel')
    Normalize_arguments.add_argument('--denominator', type=int, default=10, help='The max value to be linearly normalized')
    Normalize_arguments.add_argument('--step', type=float, default=None, help='the step size for the linear part')

    # DNA and mappability param

    DNA_arguments = parser.add_argument_group('DNA and mappability Arguments')
    DNA_arguments.add_argument('-sp',   '--species',     type=str, default = 'human', help='DNA species')
    DNA_arguments.add_argument('-DNA',  '--DNA-option',  type=str, choices=['No', 'Yes', 'dummy'], default = 'Yes', help='whether include DNA information in the dataset')

    evo2_arguments = parser.add_argument_group('Evo2 Arguments')
    evo2_arguments.add_argument('-evo2', '--evo2-option', type=str, choices=['No', 'Yes'], default = 'No', help='whether include Evo2 embedding in the dataset')

    return parser