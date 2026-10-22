import os
from utils import print_info
import argparse
from dataset.data_arguments import add_hic_dataset_arguments
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np
import torch.nn.functional as F

def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method-name', type=str)
    parser.add_argument('--debug', action='store_true')

    parser = add_hic_dataset_arguments(parser)

    #model param
    Model_arguments = parser.add_argument_group('Model Arguments')
    Model_arguments.add_argument('--input-channels', type=int, default = 1, help='The input channels of model')
    Model_arguments.add_argument('--output-channels', type=int, default = 1, help='The input channels of model')
    Model_arguments.add_argument('--dim', type=int, default = 128, help='The number of dimensions at the first layer in unet')

    Model_arguments.add_argument('--use-multiresolution-loss',  action=argparse.BooleanOptionalAction, default=True, help='Whether use multiresolution loss')
    Model_arguments.add_argument('--use-multiresolution-block', action=argparse.BooleanOptionalAction, default=True, help='Whether use multiresolution convolution')
    Model_arguments.add_argument('--relative-resolutions', type=int, nargs="*", default= [1, 2, 4, 5], help='Whether use multiresolution loss')
    Model_arguments.add_argument('--force-final-conv', action=argparse.BooleanOptionalAction, default=False, help='Whether use multiresolution loss')
    Model_arguments.add_argument('--use-mrcrossembed', action=argparse.BooleanOptionalAction, default=True, help='Whether use multiresolution loss')

    Model_arguments.add_argument('--encoder-version', type=str, default = 'v1', choices=['v1', 'v2'], help='The versino of dna encoder')
    Model_arguments.add_argument('--emb-dim', type=int, default = 128, help='The number of dimensions of the embedding')
    Model_arguments.add_argument('--normalize-emb',  action=argparse.BooleanOptionalAction, default=True, help='Whether normalize embeddings before passed to decoder')

    #train param
    Train_arguments = parser.add_argument_group('Train Arguments')
    Train_arguments.add_argument('--aux-peaks', type=str, default=None, help='name of the binned peak targets in cg_target_dir (scripts/make_aux_peaks.py) for an auxiliary 1-D BCE loss on the DNA encoder')
    Train_arguments.add_argument('--aux-weight', type=float, default=0.05, help='weight of the auxiliary peak loss')
    Train_arguments.add_argument('--reset-lr', type=float, default=None, help='on --load-checkpoint, override the restored learning rate with this constant value')
    Train_arguments.add_argument('--mixed-precision', type=str, default=None, choices=['no', 'fp16', 'bf16'], help='autocast precision for training (default: fp32)')
    Train_arguments.add_argument('--no-encoder-dropout', action='store_true', help='disable the dropout (p=0.1) inside the DNA encoder 1D conv blocks')
    Train_arguments.add_argument('--eval-only', action=argparse.BooleanOptionalAction,  default = False, help='Only evaluate the model.')
    Train_arguments.add_argument('--valid-only', type=str, default=None, help='run the validation once on the weights in this file (e.g. a BN-recalibrated checkpoint) and exit')
    Train_arguments.add_argument('--recalibrate-bn', type=str, default=None, help='recompute BatchNorm statistics of this checkpoint on training batches, save recal_<name>.pt next to it and exit')
    Train_arguments.add_argument('--bn-recal-batches', type=int, default=50, help='training batches used by --recalibrate-bn')
    Train_arguments.add_argument('--seed', type=int, default=0, help='The seed to train the model')
    Train_arguments.add_argument('--max-step', type=int, default = 50000, help='The max step of training.')
    Train_arguments.add_argument('--warmup-step', type=int, default = 0, help='the number of warm up step.')
    Train_arguments.add_argument('--lr-decay', action=argparse.BooleanOptionalAction,  default = False, help='Use cosine-decay lr.')
    Train_arguments.add_argument('--base-learning-rate', type=float, default = 1e-4, help='The initial learning rate for 1 gpu.')
    Train_arguments.add_argument('--accumulate-step', type=int, default = 1, help='step to accumulate')
    Train_arguments.add_argument('--initialize', type=str, default=None)

    Train_resuming_arguments = parser.add_argument_group('Train resuming Arguments')
    Train_resuming_arguments.add_argument('--load-checkpoint', type=str, default = None)
    Train_resuming_arguments.add_argument('--resume-wandb-run', type=str, default = None)

    #train data param
    Dataset_arguments = parser.add_argument_group('Dataset Arguments')

    Dataset_arguments.add_argument('--augment-resolution', type=int, default = None, help='read matrix in this resolution to augment')

    Dataset_arguments.add_argument('--batch-size', type=int, default = 4, help='batch size')

    Dataset_arguments.add_argument('--sample', action=argparse.BooleanOptionalAction, default=True, help='sample patches for training')
    Dataset_arguments.add_argument('--epoch-step', type=int, default = 1000, help='The step of an pseudo epoch.')

    Dataset_arguments.add_argument('--whole-row', action=argparse.BooleanOptionalAction, default=True, help='should sample include a row')
    Dataset_arguments.add_argument('--train-pos-per-row', type=int, default = 4, help='how many patches for one row.')
    Dataset_arguments.add_argument('--valid-pos-per-row', type=int, default = -1, help='how many patches for one row.')

    Dataset_arguments.add_argument('--train-hic-per-pos', type=int, default = 2, help='pair how many Hi-C to one patch')
    Dataset_arguments.add_argument('--valid-hic-per-pos', type=int, default = 8, help='pair how many Hi-C to one patch')

    Dataset_arguments.add_argument('--transpose-prob', type=float, default=0, help='The probablity to transpose matrix.')
    Dataset_arguments.add_argument('--flip-prob', type=float, default=0, help='The probablity to flip the strand.')
    
    Dataset_arguments.add_argument('--DNA-drop-prob', type=float, default=0, help='The probablity to drop DNA.')
    Dataset_arguments.add_argument('--map-drop-prob', type=float, default=0, help='The probablity to drop mappability.')

    Dataset_arguments.add_argument('--DNA-shift', type=int, default=0, help='The range to shift DNA sequence.')

    Dataset_arguments.add_argument('--target-downsample-prob', type=float, default = 0, help='target downsample rate')
    Dataset_arguments.add_argument('--target-downsample-read-count', type=float, default = 100, help='downsample target to this read count')

    return parser


class checkpoint_maintainer:
    def __init__(self, debug = False) -> None:
        self.pointing = {}
        self.debug = debug
    
    def create_checkpoint(self, file_name:str, keep = False):
        file_name = os.path.abspath(file_name)

        if file_name in self.pointing:
            print_info(f'file {file_name} already stored')
            return

        else:
            self.pointing[file_name] = set()
            if keep:
                self.pointing[file_name].add(file_name)

    def link_checkpoint(self, original_name, link_name):
        original_name = os.path.abspath(original_name)

        assert original_name in self.pointing

        for k in self.pointing.keys():
            if link_name in self.pointing[k]:
                self.pointing[k].remove(link_name)
                if not self.debug:
                    os.remove(link_name)

        if not self.debug:
            os.symlink(original_name, link_name)

        self.pointing[original_name].add(link_name)

        print_info(f'{link_name} updated.')
        
    def clean(self):
        keys = list(self.pointing.keys())
        for k in keys:
            if len(self.pointing[k]) == 0:
                os.remove(k)
                del self.pointing[k]

def cut2d(x:torch.Tensor, remove_boundary_l:int, remove_boundary_r:int= None):
    if remove_boundary_r is None:
        remove_boundary_r = remove_boundary_l
    if remove_boundary_r >0:
        x = x[..., remove_boundary_l:-remove_boundary_r, remove_boundary_l:-remove_boundary_r]
    else:
        x = x[..., remove_boundary_l:, remove_boundary_l:]
    return x

def cut1d(x:torch.Tensor, remove_boundary_l:int, remove_boundary_r:int= None):
    if remove_boundary_r is None:
        remove_boundary_r = remove_boundary_l
    if remove_boundary_r >0:
        x = x[..., remove_boundary_l:-remove_boundary_r, :]
    else:
        x = x[..., remove_boundary_l:, :]
    return x

def save_hic_batches_to_pdf(
    pred: torch.Tensor,
    target: torch.Tensor,
    position: torch.Tensor,
    vmin,
    vmax,
    save_path: str = "hic_output.pdf",
    box_size: int = None
):
    """
    Save Hi-C prediction/target batches to PDF with support for multiple channels (e.g., 1 or 2).

    Each channel of each region is visualized as a (Target | Prediction) pair.
    """

    mask = torch.isfinite(target)
    pred_m = pred * mask
    target_m = torch.nan_to_num(target) * mask
    mse = F.mse_loss(pred_m, target_m, reduction='none').mean(dim=[-2, -1]).detach().cpu().numpy()  # (B, S, R, C)

    pred_n = pred_m - pred_m.sum(dim=(-1, -2), keepdim=True) / mask.sum(dim=(-1, -2), keepdim=True).clip(min=1)
    target_n = target_m - target_m.sum(dim=(-1, -2), keepdim=True) / mask.sum(dim=(-1, -2), keepdim=True).clip(min=1)

    pred_n = (pred_n * mask).flatten(-2, -1)
    target_n = (target_n * mask).flatten(-2, -1)

    pred_n = F.normalize(pred_n, dim=-1)
    target_n = F.normalize(target_n, dim=-1)
    cos = 1 - (pred_n * target_n).sum(dim=-1).detach().cpu().numpy()  # (B, S, R, C)

    pred = pred.detach().cpu().numpy()      # (B, S, R, C, H, W)
    target = target.detach().cpu().numpy()  # (B, S, R, C, H, W)
    position = position.squeeze(2).detach().cpu().int().numpy()  # (B, S, 8)

    B, S, R, C, H, W = pred.shape

    with PdfPages(save_path) as pdf:
        for b in range(B):
            for s in range(S):
                # For each region and channel, 1 row = [Target | Prediction]
                total_plots = R * C
                n_cols = 2
                n_rows = total_plots

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
                if n_rows == 1:
                    axes = axes[None, :]
                axes = np.array(axes).reshape(n_rows, n_cols)

                # Page title
                chr1, start1, end1, strand1, chr2, start2, end2, strand2 = position[b, s]
                page_title = (
                    f"chr{chr1}:{start1:,}-{end1:,} ({'+' if strand1 == 0 else '-'})"
                    f" ↔ chr{chr2}:{start2:,}-{end2:,} ({'+' if strand2 == 0 else '-'})"
                )
                fig.suptitle(page_title, fontsize=10, y=0.995)

                for r in range(R):
                    for c in range(C):
                        row_idx = r * C + c
                        mse_val = mse[b, s, r, c]
                        cos_val = cos[b, s, r, c]

                        axes[row_idx][0].set_title(f"T{r}-C{c}", fontsize=6)
                        axes[row_idx][1].set_title(f"P{r}-C{c}", fontsize=6)

                        fig.text(
                            0.05 + 0.75,
                            1 - (row_idx + 0.5) / n_rows * 0.95,
                            f"MSE: {mse_val:.4f}\nCOS: {cos_val:.4f}",
                            ha='center', va='center', fontsize=7
                        )

                        for is_target, data, ax_col in [(True, target, 0), (False, pred, 1)]:
                            ax = axes[row_idx][ax_col]
                            im = ax.imshow(data[b, s, r, c], cmap='Reds', vmin=vmin, vmax=vmax)
                            ax.axis('off')

                            if box_size is not None:
                                h, w = data[b, s, r, c].shape
                                for y in range(0, h, box_size):
                                    for x in range(0, w, box_size):
                                        rect = Rectangle(
                                            (x - 0.5, y - 0.5), box_size, box_size,
                                            linewidth=0.5, edgecolor='white', facecolor='none'
                                        )
                                        ax.add_patch(rect)

                plt.tight_layout(rect=[0, 0, 1, 0.98], pad=0.1)
                pdf.savefig(fig)
                plt.close(fig)