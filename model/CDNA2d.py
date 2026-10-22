import torch
from torch import nn
from config import *
from utils import *
from model.CUnet import CUnet
from model.DNAencoder import DualDNAEncoder
from model.HiCencoder import HiCEncoder

class CDNA2d(nn.Module):
    # The wrapper for resolution enhancement task
    def __init__(self, decoder:nn.Module, DNA_encoder:DualDNAEncoder, HiC_encoder:HiCEncoder, normalize_emb = False):
        super().__init__()
        self.DNA_encoder = DNA_encoder
        self.HiC_encoder = HiC_encoder

        self.decoder = decoder

        self.normalize_emb = normalize_emb

    def forward(self, return_emb=False, return_emb_directly = False,input_embeds = None, DNA_embeds = None, **data):
        # shape: (B, S, H, *)
        b, s, h, *_ = data['input_matrix'].shape

        if input_embeds is None:
            input_embeds = self.HiC_encoder(data['input_matrix'].flatten(0, 2))
    
        if self.normalize_emb:
            input_embeds = nn.functional.normalize(input_embeds, dim=-3)

        input_embeds = input_embeds.unflatten(0, (b,s,h))

        data['input_matrix_embeds'] = input_embeds

        if DNA_embeds is None:
            if 'DNA_row' in data and torch.numel(data['DNA_row']) > 0:
                # have DNA input
                DNA_embeds = self.DNA_encoder(
                    data['DNA_col'], 
                    data['DNA_row'], 
                    data['mappability_col'], 
                    data['mappability_row'], 
                    return_emb=return_emb or return_emb_directly
                )
                if return_emb or return_emb_directly:
                    DNA_embeds, DNA_row_embeds = DNA_embeds

                if self.normalize_emb:
                    DNA_embeds = nn.functional.normalize(DNA_embeds, dim=-3)
            else:
                # don't have DNA input
                DNA_embeds = torch.zeros_like(input_embeds)
                DNA_row_embeds = None
        else:
            # have DNA embed input
            DNA_row_embeds = None

        data['DNA_embeds'] = DNA_embeds
        # optional auxiliary 1-D head on the DNA embeddings (Evo2HiC-cg, Orca-style peak supervision)
        aux_logits = None
        if getattr(self, 'aux_head', None) is not None and DNA_row_embeds is not None:
            aux_logits = self.aux_head(DNA_row_embeds.flatten(0, 2))      # (N, n_feat, L); returned as a 5th output

        new_data = {}

        for k, v in data.items():
            if v is None or v.numel() <= 0: continue
            _, s0, h0, *_ = v.shape
            expand_shape = [-1 for _ in v.shape]
            if s0 == 1:
                expand_shape[1] = s
            if h0 == 1:
                expand_shape[2] = h

            new_data[k] = v.expand(*expand_shape).flatten(0,2)

        if return_emb_directly:
            return input_embeds, DNA_embeds, DNA_row_embeds

        pred = self.decoder(**new_data).unflatten(0, (b, s, h))

        if return_emb:
            if aux_logits is not None:
                return pred, input_embeds, DNA_embeds, DNA_row_embeds, aux_logits
            return pred, input_embeds, DNA_embeds, DNA_row_embeds

        return pred