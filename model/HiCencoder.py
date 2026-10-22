import torch
from torch import nn
from model.multiresolution_block import ResnetBlock, CrossEmbedLayer, MRCrossEmbedLayer

class HiCEncoder(nn.Module):
    def __init__(
            self,
            init_channels,
            dim,
            init_cross_embed_kernel_sizes = (3, 7, 15),
            num_resnet = 2,
            relative_resolutions = [1],
            use_mrcrossembed = True,     # matches the training default (train_utils.py); all released checkpoints use it
            force_final_conv = True
        ):
        super(HiCEncoder, self).__init__()

        self.relative_resolutions = relative_resolutions

        if use_mrcrossembed:
            self.init_conv = MRCrossEmbedLayer(
                init_channels, 
                dim_out = dim, 
                kernel_sizes = init_cross_embed_kernel_sizes, 
                stride = 1, 
                relative_resolutions=relative_resolutions,
                force_final_conv = force_final_conv
            )
        else:
            self.init_conv = CrossEmbedLayer(init_channels, init_cross_embed_kernel_sizes, dim_out = dim, stride = 1)

        self.resnets = nn.ModuleList([ResnetBlock(dim, dim, use_gca = True, relative_resolutions=relative_resolutions, force_final_conv=force_final_conv) for _ in range(num_resnet)])

        self.final_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # input: x: (B, C, N, N), read_counts: (B) output: (B, dim, N, N)
        x = self.init_conv(x)
        for resnet in self.resnets:
            x = resnet(x)

        return self.final_conv(x)
