# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# --------------------------------------------------------

import numpy as np
from tqdm import tqdm
import torch
from config import *
from utils import *
import json
import random
from torch.utils.data import DataLoader
from dataset.hic_dna_dataset import HiC_DNA_Inference_Dataset
from dataset.hic_loader import HiC_Loader, Downsample_HiC_Loader, create_loaders_with_index
from dataset.DNA_loader import DNA_Loader
from dataset.mappability_loader import Mappability_Loader
from dataset.normalizer import Normalizer
from model.create_CDNA2d import create_model
from model.CDNA2d import CDNA2d
from scipy import sparse
from hic_utils import *
from inference.infer_utils import *

def CDNA2d_data_predict_parser():

    parser = data_predict_parser()

    Model_arguments = parser.add_argument_group('Model Arguments')
    Model_arguments.add_argument('-ckpt', '--checkpoint', type=str, required = True, help='the path to model checkpoint')

    return parser

def load_model(checkpoint):
    args_file = os.path.join(os.path.dirname(checkpoint), 'args.json')
    with open(args_file) as f:
        args = json.load(f)

    # args.json files written by train_CDNA2d_ResEnh.py use different key names and omit
    # some architecture flags; fill them in with the values used for training.
    args.setdefault('input_channels', args.get('num_channels', 1))
    args.setdefault('output_channels', args.get('num_channels', 1))
    args.setdefault('dim', args.get('unet_input_dim', 128))
    args.setdefault('force_final_conv', False)
    args.setdefault('use_mrcrossembed', True)
    args.setdefault('encoder_version', 'v1')

    normalizer = Normalizer(args['normalization'], max_reads=args['max_reads'], denominator = args['denominator'], step=args['step'])
    
    model = create_model(**{**args, 'normalizer' : normalizer, 'diffusion_steps' : 0})
    
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state_unified = {k.replace('unet', 'decoder'):v for k,v in state['model'].items()}
    # strict loading: a missing/unexpected key means the architecture does not match the checkpoint
    model.load_state_dict(state_unified, strict=True)

    return model, args, normalizer

def inference(dataset : HiC_DNA_Inference_Dataset, model:CDNA2d, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    submats = []
    submat_poses = []

    device = 'cuda'
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader)
        for data in pbar:
            pbar.set_description(
                desc=f"[Predicting in valid set]")
            
            for k in data:
                data[k] = data[k].to(device)
                
            sr_n = model(**data).flatten(0,2)
            sr_n = sr_n[:, 0, :, :].cpu().numpy()

            for i in range(sr_n.shape[0]):
                sr_un = dataset.normalizer.unnormalize(sr_n[i]).clip(min=0)
                submats.append(sparse.coo_matrix(sr_un))

            submat_poses.extend(data['positions'].flatten(0,2).tolist())
        
    return submats, submat_poses

if __name__ == '__main__':
    parser = CDNA2d_data_predict_parser()
    parser.set_defaults(
        whole_row = False
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
    downsample = args.downsample
    resolution = args.resolution
    norm = args.norm
    read_count = modelargs['read_count']
    max_seperation = args.max_seperation
    DNA_option = args.DNA_option

    target_resolution = args.target_resolution

    if species in splits.keys():
        input_file = args.input_file

        if input_file is not None:
            if not os.path.isfile(input_file):
                input_file = os.path.join(hic_data_dir, input_file)
            if args.input_option == 'expected':
                loader = HiC_Loader(input_file, resolution=resolution, norm=norm, read_count=read_count)
            else:
                loader = Downsample_HiC_Loader(input_file, downsample, species, resolution=resolution, norm=norm, read_count=read_count)
            accession = '.'.join(os.path.basename(input_file).split('.')[:-1])
            hic_loaders = [(accession, loader)]
        else:
            hic_loaders = create_loaders_with_index('test', species, return_name=True, downsample=downsample, resolution=resolution, norm=norm, read_count=read_count)

        DNA_loader = DNA_Loader(DNA_map[species], DNA_option)
        mappability_loader = Mappability_Loader(mappability_map[species], DNA_option)

        for name, hic_loader in hic_loaders:

            fname = f'{name}_enhanced_{args.split}.hic'
            output = os.path.join(out_dir, fname)

            if os.path.exists(output):
                print(f'{output} already exists. Skip.')
                continue

            print(f'Saving the data to {output}')

            first = True
            for ch in splits[args.species][args.split]:
                split = (ch, )

                test_set = HiC_DNA_Inference_Dataset(
                    chromosome_split = split,
                    HiC_loader = [hic_loader],
                    DNA_loader = DNA_loader,
                    Mappability_loader = mappability_loader,
                    normalizer = normalizer,
                    **args.__dict__
                )
                    
                submats, submat_poses = inference(test_set, model, args.batch_size)

                chrom_size = DNA_loader.get_size(ch)

                hic = construct_hic_matrices(submats, submat_poses, ch, ch, chrom_size, chrom_size, resolution, target_resolution, max_seperation)
                
                datas = {(id2chr(ch), id2chr(ch)): hic}
            
                save_hic(datas, None, output, first = first, finished=False)
                first = False

                del datas

            if species in references:
                reference = references[species]
            else:
                reference = output.replace('.hic', '.chrom.size')
                with open(reference, 'w') as f:
                    for ch in hic_loader.chromosomes:
                        if ch.name in hic_loader.chromosome_filter:continue
                        f.write(f"{id2chr(ch.name)}\t{ch.length}\n")
                
            save_hic({}, reference, output, target_resolution, first=False, finished=True)

    elif species == 'multi':
        data_index = args.data_index

        from dataset.multi_species import create_multispecies_loaders_with_index
        dataiter = create_multispecies_loaders_with_index(
            data_index = data_index
        )

        for species, hic_file, dna_file in dataiter:
            fname = f'{species}_enhanced.hic'
            output = os.path.join(out_dir, fname)

            if os.path.exists(output):
                print(f'{output} already exists. Skip.')
                continue

            dna_loader = DNA_Loader(dna_file, option = DNA_option, encoding_dir=dnazoo_encoding_dir)
            mappability_loader = Mappability_Loader(None, 'dummy')
            hic_loader = HiC_Loader(
                            hic_file,
                            resolution=resolution,
                            read_count=read_count,
                            chr_pos=dna_loader.chr_pos
                        ) if downsample is None else\
                        Downsample_HiC_Loader(
                            hic_file, 
                            downsample, 
                            species, 
                            resolution=resolution,
                            read_count=read_count,
                            chr_pos=dna_loader.chr_pos
                        )

            reference = output.replace('.hic', '.chrom.size')
            ref_file = open(reference, 'w')

            first = True
            for i, ch in enumerate(dna_loader.chr_pos):
                split = (i, )

                test_set = HiC_DNA_Inference_Dataset(
                    chromosome_split=split,
                    Downsampled_HiC_loader = hic_loader,
                    DNA_loader = dna_loader,
                    Mappability_loader = mappability_loader,
                    normalizer = normalizer,
                    **args.__dict__
                )
                    
                submats, submat_poses = inference(test_set, model, args.batch_size)

                chrom_size = dna_loader.get_size(i)

                hic = construct_hic_matrices(submats, submat_poses, i, i, chrom_size, chrom_size, resolution, target_resolution, max_seperation)
                
                datas = {(ch, ch): hic}
            
                save_hic(datas, None, output, first = first, finished=False)
                first = False

                ref_file.write(f"{ch}\t{dna_loader.get_size(ch)}\n")

                del datas, submats, submat_poses
                break

            del hic_loader, dna_loader, mappability_loader

            ref_file.close()

            save_hic({}, reference, output, target_resolution, first = False, finished=True)