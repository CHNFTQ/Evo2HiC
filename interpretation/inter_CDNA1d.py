# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# script to interpret track prediction
# --------------------------------------------------------

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from config import *
from utils import *
import json
import random
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import pandas as pd
import captum
from tangermeme.deep_lift_shap import deep_lift_shap, dinucleotide_shuffle

from dataset.hic_dna_track_dataset import HiC_DNA_track_Dataset, create_hic2track_loaders
from dataset.hic_loader import HiC_Loader
from dataset.DNA_loader import DNA_Loader
from dataset.mappability_loader import Mappability_Loader
from model.CDNA1d import CDNA1d
from hic_utils import *
from inference.infer_utils import data_predict_parser, construct_1d_tracks
from inference.inference_CDNA1d import *

def CDNA1d_data_interpret_parser():

    parser = data_predict_parser()

    Model_arguments = parser.add_argument_group('Model Arguments')
    Model_arguments.add_argument('-ckpt', '--checkpoint', type=str, required = True, help='the path to model checkpoint')

    Inter_arguments = parser.add_argument_group('Interpret Arguments')
    Inter_arguments.add_argument('-m', '--method', type=str, default='deepliftshap', choices=['deepliftshap', 'IG', 'perturb'], help='the method to interpret')

    return parser

def interpret(method, dataset : HiC_DNA_track_Dataset, model:CDNA1d, target:int, batch_size:int):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    device = 'cuda'

    class wrapper(nn.Module):
        def __init__(self, model:CDNA1d, return_mean = False, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = model
            self.return_mean = return_mean
        def forward(self, DNA, mappability, input_matrix):
            pred : torch.Tensor = self.model(
                DNA0 = DNA.unsqueeze(1).unsqueeze(2).transpose(-1,-2), 
                mappability0 = mappability, 
                input_matrix = input_matrix, 
            ).flatten(0,2)

            if self.return_mean:
                s = pred.mean(dim=-1)
                return s
            else:
                return pred
    
    model_wrapper = wrapper(model, return_mean = (method != 'perturb'))
    model_wrapper = model_wrapper.to(device)

    seqs = []
    attrs = []
    poses = []

    model_wrapper.eval()

    if method == 'IG':
        IG = captum.attr.IntegratedGradients(model_wrapper)

    pbar = tqdm(loader)
    for data in pbar:
        DNA0, map0, hic = data['DNA0'].to(device), data['mappability0'].to(device), data['input_matrix'].to(device)
        DNA0: torch.Tensor

        inputs = DNA0.flatten(0,2).transpose(-1,-2)

        if method == 'deepliftshap':
            if not torch.all(inputs.sum(axis=1) == 1):
                continue
            else:
                attribution = deep_lift_shap(
                    model_wrapper, 
                    inputs, 
                    args=(map0, hic), 
                    target = target,
                    batch_size=2,
                    random_state=0,
                    warning_threshold = 5e-2
                )
        elif method == 'IG':
            n_shuffles = 20
            if not torch.all(inputs.sum(axis=1) == 1):
                continue
            else:
                baselines = dinucleotide_shuffle(inputs.cpu(), n=n_shuffles, random_state=0).to(device)
                attris = []
                deltas = []
                for i in tqdm(range(n_shuffles), leave=False):
                    attr, delta = IG.attribute(
                        inputs, 
                        baselines[:, i, :, :], 
                        target = target,
                        additional_forward_args = (map0, hic), 
                        n_steps = 50,
                        internal_batch_size = 4,
                        return_convergence_delta = True
                    )
                    attr_s = attr.detach().sum(dim=-2, keepdim=True) * inputs

                    attris.append(attr_s)
                    deltas.append(delta.cpu().item())

                attribution = torch.stack(attris, dim=-1).mean(dim=-1)

                pbar.set_description(
                    desc=f"[delta={np.mean(deltas)}]")
        elif method == 'perturb':
            raise NotImplementedError
        
        seqs.append(inputs.cpu().numpy())
        attrs.append(attribution.cpu().numpy())
        poses.append(data['positions'].flatten(0,2).cpu().numpy())

    seqs = np.concatenate(seqs, axis=0)
    attrs = np.concatenate(attrs, axis=0)
    poses = np.concatenate(poses, axis=0)

    return seqs, attrs, poses

if __name__ == '__main__':
    parser = CDNA1d_data_interpret_parser()
    parser.set_defaults(
        batch_size=1
    )
    args = parser.parse_args()
   
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    try:
        ckpt = os.readlink(args.checkpoint)
    except:
        ckpt = args.checkpoint

    save_dir = args.save_dir
    print(f'Inferencing {save_dir}')
    out_dir = os.path.join('.'.join(ckpt.split('.')[:-1]), save_dir)
    mkdir(out_dir)

    with open(os.path.join(out_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    model, modelargs, normalizer = load_model(ckpt)

    species = args.species
    resolution = args.resolution
    read_count = modelargs['read_count']
    max_separation = args.max_separation
    DNA_option = args.DNA_option

    if species in splits.keys():
        assert species == 'human'
        names, hic_loaders, track_loaders = create_hic2track_loaders(None, resolution=resolution, return_name=True, read_count=read_count)

        DNA_loader = DNA_Loader(DNA_map[species], DNA_option)
        mappability_loader = Mappability_Loader(mappability_map[species], DNA_option)

        result_file = os.path.join(out_dir, 'result.tsv')

        results = {'Name' : [], 'Chr': []}
        for track_name in tracks.keys():
            results[track_name] = []

        for name, hic_loader, track_loader in zip(names, hic_loaders, track_loaders):
            cellline_dir = os.path.join(out_dir, name)
            mkdir(cellline_dir)

            print(f'Saving the data to {cellline_dir}')

            split = splits[args.species][args.split]

            test_set = HiC_DNA_track_Dataset(
                chromosome_split=split,
                HiC_loaders = [hic_loader],
                Track_loaders = [track_loader],
                DNA_loader = DNA_loader,
                Mappability_loader= mappability_loader,
                normalizer=normalizer,
                training=False,
                hic_per_pos=1,
                **args.__dict__
            )

            for ch in split:
                chrom_size = DNA_loader.get_size(ch)
                seq = np.expand_dims(DNA_loader.get(ch, 0, chrom_size, 0).transpose(-1, -2), 0)
                print(seq.shape)

                seq_file = os.path.join(cellline_dir, f'seq_{ch}.npy')
                np.save(seq_file, seq)

            for i, track_name in enumerate(tracks.keys()):
                cur_dir = os.path.join(cellline_dir, track_name)
                mkdir(cur_dir)

                seqs, attrs, attr_poses = interpret(args.method, test_set, model, i, args.batch_size)

                seq_s_file = os.path.join(cellline_dir, f'seq_sample.npy')
                np.save(seq_s_file, seqs)

                attr_s_file = os.path.join(cur_dir, f'attr_sample.npy')
                np.save(attr_s_file, attrs)

                pos_s_file = os.path.join(cur_dir, f'pos_sample.npy')
                np.save(pos_s_file, attr_poses)

                motif_file = os.path.join(cur_dir, f'modisco_results.h5')
                window_size = attrs.shape[-1]
                motif_find = f'modisco motifs -s {seq_s_file} -a {attr_s_file} -n 20000 -w {window_size} -o {motif_file}'
                print(motif_find)
                os.system(motif_find)

                report_dir = os.path.join(cur_dir, f'report')
                motif_report = f'modisco report -i {motif_file} -m {motif_data} -o {report_dir}'
                print(motif_report)
                os.system(motif_report)

                for ch in split:
                    t_attr = construct_1d_tracks(attrs, attr_poses, ch, chrom_size, 1, 4)
                    t_attr = np.expand_dims(t_attr, 0)

                    attr_file = os.path.join(cur_dir, f'attr_{ch}.npy')
                    np.save(attr_file, t_attr)