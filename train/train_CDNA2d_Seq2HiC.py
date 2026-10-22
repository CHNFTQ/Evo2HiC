# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# --------------------------------------------------------

import os
import time
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import logging
import json
import random
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR

from dataset.hic_dna_dataset import prepare_datasets
from dataset.normalizer import Normalizer
from model.create_CDNA2d import create_model, create_loss
from model.Evoencoder import DualEvoEncoder

from config import *
import utils
from utils import *
from train.train_utils import *

torch.autograd.set_detect_anomaly(True)
import argparse

def recalibrate_bn(module, loader, n_batches, evo_encoder=None, args=None):
    """Recompute the BatchNorm running statistics of `module` on n_batches training batches (cumulative
    average, momentum=None). Mirrors the training forward call."""
    bns = [m for m in module.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    moms = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None
    was_training = module.training; module.train()
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= n_batches:
                break
            data.pop('target_1d', None)
            if args is not None and args.evo2_option == 'Yes' and args.use_evo_for_DNA:
                Evo_embeds = evo_encoder(data['embedding_col'], data['embedding_row'], data['mappability_col'], data['mappability_row'])
                module(**data, DNA_embeds=Evo_embeds)
            else:
                module(**data, return_emb=True)
    for m, mom in zip(bns, moms):
        m.momentum = mom
    module.train(was_training)


def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps= args.accumulate_step,
        kwargs_handlers =  [DistributedDataParallelKwargs(find_unused_parameters = True)],
        **({'mixed_precision': args.mixed_precision} if args.mixed_precision else {})
    )

    utils.VERBOSE = accelerator.is_main_process

    random.seed(args.seed + accelerator.process_index)
    np.random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)
    torch.cuda.manual_seed(args.seed + accelerator.process_index)

    datestr = time.strftime('%m_%d_%H_%M')

    # out_dir: directory storing checkpoint files
    save_name = f'{datestr}_{args.method_name}_{args.resolution}'
    out_dir = os.path.join(save_dir, save_name)

    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)

        log_file_path = os.path.join(out_dir, "experiment.log")

        with open(os.path.join(out_dir, "args.json"), 'w') as f:
            json.dump(args.__dict__, f, indent=2, default=float)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            filename=log_file_path, 
            filemode='w', 
            force=True
        )

        vis_path = os.path.join(out_dir, 'vis')
        mkdir(vis_path)

        if not args.debug:
            if args.resume_wandb_run is not None:
                wandb.init(
                    project = "Cdiffusion",
                    id = args.resume_wandb_run,
                    resume = 'must',

                    config = args.__dict__
                )
            else:
                wandb.init(
                    project = "Cdiffusion",

                    config = args.__dict__
                )

    normalizer = Normalizer(args.normalization, max_reads=args.max_reads, denominator = args.denominator, step=args.step)

    model = create_model(**{**args.__dict__, 'normalizer' : normalizer, 'diffusion_steps' : 0})
    if args.aux_peaks is not None:
        from dataset.peak_loader import PeakLoader
        peak_loader = PeakLoader(args.aux_peaks, resolution=args.resolution)
        model.aux_head = nn.Conv1d(model.DNA_encoder.encoder.dim, peak_loader.n_feat, 1)
        print_info(f'auxiliary peak targets: {args.aux_peaks}, {peak_loader.n_feat} features, weight {args.aux_weight}')

    if args.evo2_option == 'Yes':
        evo_encoder = DualEvoEncoder(
            resolution = args.resolution, 
            input_dim = evo2_hidden_size * 2,
            dim = args.emb_dim,
            relative_resolutions=args.relative_resolutions,
        )

    step = 0

    optimizer = AdamW(
                [
                    {'params': model.parameters(), 'lr' : args.base_learning_rate},
                ] + ([
                    {'params': evo_encoder.parameters(), 'lr' : args.base_learning_rate},
                ] if args.evo2_option == 'Yes' else []),
                betas = (0.9, 0.95)
            )

    max_step = args.max_step
    warmup_step = args.warmup_step
    mm = max_step*accelerator.num_processes
    wm = warmup_step*accelerator.num_processes

    if warmup_step>0:
        warmup_scheduler = LinearLR(optimizer, start_factor=1/wm, end_factor=1.0, total_iters=wm)
        scheduler = CosineAnnealingLR(optimizer, T_max=mm - wm) if args.lr_decay else ConstantLR(optimizer, factor=1, total_iters=10000000)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[wm])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=mm) if args.lr_decay else ConstantLR(optimizer, factor=1, total_iters=10000000)

    if accelerator.is_main_process:
        best = {}

    if args.load_checkpoint is not None:
        checkpoint = args.load_checkpoint
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)

        model.load_state_dict(state['model'])
        step = state['step']
        optimizer.load_state_dict(state['optimizer'])
        if args.reset_lr is None:
            scheduler.load_state_dict(state['scheduler'])

        if args.evo2_option == 'Yes':
            if 'evo_encoder' in state:
                evo_encoder.load_state_dict(state['evo_encoder'])

        if accelerator.is_main_process:
            with open(os.path.join(os.path.dirname(checkpoint), 'best.json')) as f:
                best = json.load(f)

        if args.reset_lr is not None:
            # continue from a checkpoint at a new constant learning rate (the restored optimizer state would
            # otherwise put the learning rate back to its old value)
            for _g in optimizer.param_groups:
                _g['lr'] = args.reset_lr; _g['initial_lr'] = args.reset_lr
            scheduler = ConstantLR(optimizer, factor=1, total_iters=10000000)
            print_info(f'learning rate reset to {args.reset_lr} (constant) for this continuation')

        print_info('loaded')
    elif args.initialize is not None:
        checkpoint = args.initialize
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state_unified = {k.replace('unet', 'decoder'):v for k,v in state['model'].items()}
        missing, unexpected = model.load_state_dict(state_unified, strict=False)
        assert not unexpected and all(k.startswith('aux_head') for k in missing), (missing, unexpected)   # only a new aux head may be missing

        if args.evo2_option == 'Yes':
            if 'evo_encoder' in state:
                evo_encoder.load_state_dict(state['evo_encoder'])

        print_info('model initialized')
    else:
        if args.initialize_evo_encoder is not None:
            state1 = torch.load(args.initialize_evo_encoder, map_location='cpu', weights_only=True)
            assert 'evo_encoder' in state1
            evo_encoder.load_state_dict(state1['evo_encoder'])

            print_info('evo_encoder initialized')

        if args.initialize_dna_encoder is not None:
            state1 = torch.load(args.initialize_dna_encoder, map_location='cpu', weights_only=True)
            dna_encoder_state_dict = {
                k.replace('DNA_encoder.', ''): v
                for k, v in state1['model'].items()
                if k.startswith('DNA_encoder.')
            }

            model.DNA_encoder.load_state_dict(dna_encoder_state_dict)

            print_info('DNA_encoder initialized')

    if args.no_encoder_dropout:
        _nd = 0
        for _m in model.DNA_encoder.modules():
            if isinstance(_m, torch.nn.Dropout):
                _m.p = 0.0; _nd += 1
        print_info(f'encoder dropout disabled in {_nd} modules')

    if args.evo2_option == 'Yes' and not args.use_evo_for_DNA:
        for param in evo_encoder.parameters():
            param.requires_grad = False

    accelerator.wait_for_everyone()

    if args.evo2_option == 'Yes':
        model, evo_encoder, optimizer, scheduler = accelerator.prepare(model, evo_encoder, optimizer, scheduler)
    else:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    train_set, valid_set, _ = prepare_datasets(**{**args.__dict__, 'normalizer' : normalizer, 'dataset_size' : accelerator.num_processes * args.epoch_step * args.batch_size})
    if args.aux_peaks is not None:
        train_set.peak_loader = peak_loader
        valid_set.peak_loader = peak_loader

    train_dl = DataLoader(train_set, batch_size = args.batch_size, shuffle=True,  num_workers = 20, drop_last=True)
    valid_dl = DataLoader(valid_set, batch_size = args.batch_size, shuffle=False, num_workers = 20)

    train_dl, valid_dl = accelerator.prepare(train_dl, valid_dl)

    if args.recalibrate_bn is not None:
        # recompute BatchNorm statistics of the given checkpoint on training batches, save recal_<name>.pt and exit
        _w = torch.load(args.recalibrate_bn, map_location='cpu', weights_only=True)
        _m = accelerator.unwrap_model(model)
        _mk, _uk = _m.load_state_dict({k.replace('unet', 'decoder'): v for k, v in _w['model'].items()}, strict=False); print_info(f'recal load: {len(_mk)} missing, {len(_uk)} unexpected keys'); assert not _uk and all(k.startswith('aux_head') for k in _mk), (_mk, _uk)
        recalibrate_bn(_m, train_dl, args.bn_recal_batches, evo_encoder if args.evo2_option == 'Yes' else None, args)
        _out = os.path.join(os.path.dirname(args.recalibrate_bn), 'recal_' + os.path.basename(args.recalibrate_bn))
        torch.save({'model': _m.state_dict(), 'step': _w.get('step', -1), 'bn_recal_batches': args.bn_recal_batches}, _out)
        print_info(f'BN-recalibrated weights written to {_out}'); return

    if args.valid_only is not None:
        # validation of the given weights (e.g. a BN-recalibrated checkpoint) with the same MSE/PCC as the training loop
        _w = torch.load(args.valid_only, map_location='cpu', weights_only=True)
        _m = accelerator.unwrap_model(model)
        _mk, _uk = _m.load_state_dict({k.replace('unet', 'decoder'): v for k, v in _w['model'].items()}, strict=False)
        assert not _uk and all(k.startswith('aux_head') for k in _mk), (_mk, _uk)
        model.eval()
        _mses, _t, _p = [], [], []
        with torch.no_grad():
            for data in tqdm(valid_dl, disable=not accelerator.is_main_process):
                data.pop('target_1d', None)
                pred = model(**data, return_emb=True)[0]
                pred = (pred + pred.transpose(-1, -2)) / 2
                target = data['target_matrix']; mask = torch.isfinite(target)
                _mses.append(F.mse_loss(pred * mask, torch.nan_to_num(target) * mask, reduction='none').mean())
                _t.append(target); _p.append(pred)
        _mses = accelerator.gather(torch.stack(_mses)); _t = accelerator.gather(torch.concat(_t)); _p = accelerator.gather(torch.concat(_p))
        if accelerator.is_main_process:
            _t, _p = _t.cpu().numpy(), _p.cpu().numpy(); pccs = []
            for k in range(1, _t.shape[-1]):
                tk = np.diagonal(_t[..., 0, :, :], offset=k, axis1=-2, axis2=-1); pk = np.diagonal(_p[..., 0, :, :], offset=k, axis1=-2, axis2=-1)
                m = np.isfinite(tk); pccs.append(pearsonr(tk[m], pk[m])[0])
            print_info(f'VALID-ONLY\t{args.valid_only}\tstep={_w.get("step", -1)}\tmse={_mses.mean().item():.8f}\tpcc={np.mean(pccs):.6f}')
        return

    if accelerator.is_main_process:
        ckpt_maintainer = checkpoint_maintainer(debug = args.debug)

        print_info('Start training.')

    train_losses = []

    epoch = int(np.ceil((max_step-step) / len(train_dl)))
    for e in range(epoch):

        model.train()
        if args.evo2_option == 'Yes':
            evo_encoder.train()

        pbar = tqdm(train_dl, disable = not accelerator.is_main_process)
        for data in pbar:
            with accelerator.accumulate(model):
                target_1d = data.pop('target_1d', None)
                aux_logits = None
                if args.evo2_option == 'Yes' and args.use_evo_for_DNA:
                    Evo_embeds = evo_encoder(data['embedding_col'], data['embedding_row'], data['mappability_col'], data['mappability_row'])
                    pred = model(**data, DNA_embeds = Evo_embeds)
                else:
                    out = model(**data, return_emb = True)
                    pred, _, _, DNA_row_embeds = out[:4]
                    aux_logits = out[4] if len(out) > 4 else None

                pred = (pred + pred.transpose(-1, -2))/2
                target = data['target_matrix']

                mask = torch.isfinite(target)
                pred_m = pred*mask
                target_m = torch.nan_to_num(target)*mask
                mse = F.mse_loss(pred_m, target_m, reduction='none').mean()

                loss = mse
                if aux_logits is not None and target_1d is not None and target_1d.numel() > 0:
                    aux_loss = F.binary_cross_entropy_with_logits(aux_logits, target_1d.reshape(-1, *target_1d.shape[-2:]))
                    loss = mse + args.aux_weight * aux_loss

                if args.evo2_option == 'Yes' and not args.use_evo_for_DNA:
                    _, Evo_row_embeds = evo_encoder(data['embedding_col'], data['embedding_row'], data['mappability_col'], data['mappability_row'], return_emb=True)

                    DNA_row_embeds = DNA_row_embeds.transpose(-1, -2)
                    Evo_row_embeds = Evo_row_embeds.transpose(-1, -2)

                    loss1 = 1-(nn.functional.normalize(DNA_row_embeds, dim=-1) * nn.functional.normalize(Evo_row_embeds, dim=-1)).sum(dim=-1).mean()

                    loss = loss + args.DNA_evo2_weight * loss1

                accelerator.backward(loss)

                if accelerator.is_main_process and step % 500 == 0:
                    vmin, vmax = normalizer.normalize(0), normalizer.normalize(100)
                    save_hic_batches_to_pdf(
                        pred,
                        target, 
                        data['positions'], 
                        save_path=os.path.join(vis_path, f'vis_{step}.pdf'), 
                        vmin=vmin,
                        vmax=vmax
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1

            train_losses.append(loss.cpu().item())

            learning_rate = scheduler.get_last_lr()[0]

            if accelerator.is_main_process and not args.debug:
                results = {
                    'train loss (DNA-HiC)' : mse.cpu().item(),
                    **({'train loss (aux-1d)': aux_loss.cpu().item()} if args.aux_peaks is not None else {}),
                    'train loss (overall)' : loss.cpu().item(),
                    'learning rate' : learning_rate
                }
                if args.evo2_option == 'Yes' and not args.use_evo_for_DNA:
                    results.update(
                        {
                            'train loss (DNA-evo2)' : loss1.cpu().item(),
                        }
                    )
                wandb.log(
                    results,
                    step = step
                )

            if accelerator.is_main_process:
                pbar.set_description(f'Epoch: {e}/{epoch}, Step: {step}, Avg loss: { np.mean(train_losses):.4e}, current loss: {np.mean(train_losses[-args.accumulate_step:]):.4e}')

        #start eval
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        model.eval()
        if args.evo2_option == 'Yes':
            evo_encoder.eval()

        if accelerator.is_main_process:
            ckpt = os.path.join(out_dir, f'{step}.pt')
            state = {
                'model' : accelerator.unwrap_model(model).state_dict(),
                'step' : step,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }
            if args.evo2_option == 'Yes':
                state.update(
                    {
                        'evo_encoder' : accelerator.unwrap_model(evo_encoder).state_dict(),
                    }
                )

            torch.save(state, ckpt)

            ckpt_maintainer.create_checkpoint(ckpt, keep = (step % 10000 == 0))

            ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'last.pt'))

            print_info(("Step: %d, \t Train, \t LR: %0.8f, \t Loss: %0.8f" % (step, learning_rate, np.mean(train_losses))))

            train_losses = []

        results = {}
        def val(dl, key_suffix = ''):
            with torch.no_grad():
                valid_bar = tqdm(dl, disable = not accelerator.is_main_process)
                targets = []
                preds = []
                valid_mses = []
                valid_loss1es = []
                valid_losses = []
                first = True
                for data in valid_bar:
                    data.pop('target_1d', None)
                    if args.evo2_option == 'Yes' and args.use_evo_for_DNA:
                        Evo_embeds = evo_encoder(data['embedding_col'], data['embedding_row'], data['mappability_col'], data['mappability_row'])
                        pred = model(**data, DNA_embeds = Evo_embeds)
                    else:
                        pred, HiC_embeds, DNA_embeds, DNA_row_embeds = model(**data, return_emb = True)[:4]

                    pred = (pred + pred.transpose(-1, -2))/2
                    target = data['target_matrix']

                    if accelerator.is_main_process and first:
                        vmin, vmax = normalizer.normalize(0), normalizer.normalize(100)
                        save_hic_batches_to_pdf(
                            pred,
                            target, 
                            data['positions'], 
                            save_path=os.path.join(vis_path, f'vis_val_{step}.pdf'), 
                            vmin=vmin, 
                            vmax=vmax
                        )
                    if first:
                        accelerator.wait_for_everyone()
                    first = False

                    mask = torch.isfinite(target)
                    pred_m = pred*mask
                    target_m = torch.nan_to_num(target)*mask
                    mse = F.mse_loss(pred_m, target_m, reduction='none').mean()

                    loss = mse

                    targets.append(target)
                    preds.append(pred)

                    valid_mses.append(mse)

                    if args.evo2_option == 'Yes' and not args.use_evo_for_DNA:
                        _, Evo_row_embeds = evo_encoder(data['embedding_col'], data['embedding_row'], data['mappability_col'], data['mappability_row'], return_emb=True)

                        DNA_row_embeds = DNA_row_embeds.transpose(-1, -2)
                        Evo_row_embeds = Evo_row_embeds.transpose(-1, -2)

                        loss1 = 1-(nn.functional.normalize(DNA_row_embeds, dim=-1) * nn.functional.normalize(Evo_row_embeds, dim=-1)).sum(dim=-1).mean()

                        valid_loss1es.append(loss1)

                        loss = loss + args.DNA_evo2_weight * loss1

                    valid_losses.append(loss)

            targets = torch.concat(targets)
            preds = torch.concat(preds)

            valid_mses = torch.stack(valid_mses)
            valid_losses = torch.stack(valid_losses)

            targets = accelerator.gather(targets)
            preds = accelerator.gather(preds)

            valid_mses = accelerator.gather(valid_mses)
            valid_losses = accelerator.gather(valid_losses)

            if args.evo2_option == 'Yes' and not args.use_evo_for_DNA:
                valid_loss1es = torch.stack(valid_loss1es)
                valid_loss1es = accelerator.gather(valid_loss1es)

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                targets = targets.cpu().numpy()
                preds = preds.cpu().numpy()
                C = targets.shape[-3]
                for c in range(C):
                    pccs = []
                    spcs = []
                    for k in range(1, targets.shape[-1]):
                        pk = np.diagonal(preds[..., c, :, :], offset=k, axis1=-2, axis2=-1)
                        tk = np.diagonal(targets[..., c, :, :], offset=k, axis1=-2, axis2=-1)
                        mask = np.isfinite(tk)
                        t, p = tk[mask], pk[mask]
                        pcc = pearsonr(t, p)[0]
                        spc = spearmanr(t, p)[0]
                        pccs.append(pcc)
                        spcs.append(spc)

                    pcc = np.mean(pccs)
                    spc = np.mean(spcs)

                    kpcc = 'PCC' + f'({args.target_option.split("+")[c]})' + key_suffix
                    results[kpcc] = pcc
                    print_info(f"Step: {step}, \t valid {kpcc} is {pcc}")
                    if pcc > best.get(kpcc, -1):
                        best[kpcc] = pcc
                        ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'best_{kpcc}.pt'))

                    kspc = 'SPC' + f'({args.target_option.split("+")[c]})' + key_suffix
                    results[kspc] = spc
                    print_info(f"Step: {step}, \t valid {kspc} is {spc}")
                    if spc > best.get(kspc, -1):
                        best[kspc] = spc
                        ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'best_{kspc}.pt'))

                valid_mse = valid_mses.mean().item()
                k = 'valid loss (mse)'
                results[k] = valid_mse
                print_info(f"Step: {step}, \t {k} is {valid_mse}")
                if valid_mse < best.get(k, 100):
                    best[k] = valid_mse
                    ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'best_valid_loss_mse.pt'))

                valid_loss = valid_losses.mean().item()
                k = 'valid loss (sum)'
                results[k] = valid_loss
                print_info(f"Step: {step}, \t {k} is {valid_loss}")
                if valid_loss < best.get(k, 100):
                    best[k] = valid_loss
                    ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'best_valid_loss.pt'))

                if args.evo2_option == 'Yes' and not args.use_evo_for_DNA:
                    valid_loss1 = valid_loss1es.mean().item()
                    k = 'valid loss (DNA-evo2)'
                    results[k] = valid_loss1
                    print_info(f"Step: {step}, \t {k} is {valid_loss1}")
                    if valid_loss1 < best.get(k, 100):
                        best[k] = valid_loss1
                        ckpt_maintainer.link_checkpoint(ckpt, os.path.join(out_dir, f'best_valid_loss_DNAEvo2.pt'))

        val(valid_dl)

        if accelerator.is_main_process:
            with open(os.path.join(out_dir, "best.json"), 'w') as f:
                json.dump(best, f, indent=2, default=float)
            ckpt_maintainer.clean()

            if not args.debug:
                wandb.log(
                    results,
                    step=step
                )

        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == '__main__':
    parser = train_parser()
    parser.add_argument('--use-evo-for-DNA', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--initialize-evo-encoder', type=str, default=None)
    parser.add_argument('--initialize-dna-encoder', type=str, default=None)
    parser.set_defaults(
        method_name = 'CDNA2d-Seq2HiC',

        resolution = 4000,

        input_option = 'expected',
        target_option = 'norm',
        target_specified = '4DNFI2TK7L2F',
        max_separation = 0,

        chunk = 280,
        stride = 125,

        augment_resolution = 2000,

        batch_size = 7,
        sample = True,
        whole_row = False,
        train_pos_per_row = 1,
        valid_pos_per_row = 1,
        train_hic_per_pos = -1,
        valid_hic_per_pos = -1,

        flip_prob = 0.5,
        DNA_shift = 100,

        max_step = 20000
    )
    args = parser.parse_args()

    train(args)