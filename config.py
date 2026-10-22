# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# The dataset settings for experiments.
# --------------------------------------------------------

# config for pretrain and resolution enhancemnet
hic_data_dir = 'data/raw_hic'
hic_property_dir = 'data/hic/property'
hic_downsample_dir = 'data/hic/downsample'
hic_loop_dir = 'data/hic/hiccups_loop'
hic_index = 'data/hic_index.tsv'
hic_hc_index = 'data/hic_highcoverage.tsv'

# config for dnazoo
dnazoo_data_dir = 'data/multi_species'
dnazoo_hic_dir = 'data/multi_species/raw_hic_fixed'
dnazoo_fasta_dir = 'data/multi_species/raw_fasta'
dnazoo_encoding_dir = 'data/dna/multispecies'
dnazoo_index = 'data/species_list_rawchrom.txt'
dnazoo_tool_path = 'data/3d-dna'
dnazoo_downsample_dir = 'data/hic/downsample_multi'

motif_data = 'data/dna/motifs.meme'

# config for predicting epigenomics
hic2tarck_dir = 'data/hic2track'
hic2track_index = 'data/hic2track/index.tsv'

# config for Evo2HiC-cg (Seq2HiC with an Orca-style coarse-grained target, see scripts/make_cg_targets.py
# and scripts/make_aux_peaks.py)
cg_target_dir = 'data/cg_targets'

references = {
    'human' : 'hg38',
    'mouse' : 'mm10'
}

DNA_map = {
    'human' : 'data/dna/human/hg38.fa',
    'mouse' : 'data/dna/mouse/mm10.fa',
    'zebrafish' : 'data/dna/zebrafish/danRer11.fa'
}
mappability_map = {
    'human' : 'data/dna/human/k100.Umap.MultiTrackMappability.bw',
    'mouse' : 'data/dna/mouse/k100.Umap.MultiTrackMappability.bw',
    'zebrafish' : None
}
evo2_embedding_map = {
    'human': 'data/dna/human/hg38_2000_evo2_7b',
    'mouse': 'data/dna/mouse/mm10_evo2_7b/'
}
evo2_hidden_size = 4096

tracks = {
    'DNase': 'DNase.bw', 
    'CTCF': 'CTCF.bw', 
    'H3K27ac': 'H3K27ac.bw',
    'H3K27me3': 'H3K27me3.bw',
    'H3K4me3': 'H3K4me3.bw'
}

juicer_tool_path = 'misc/juicer_tools.2.20.00.jar'

save_dir = 'checkpoints'

#Data configuration

res_list = [
    1000, 2000, 4000,
    5_000, 10_000, 25_000, 50_000, 100_000, 
    250_000, 500_000, 1_000_000
]

# 'train' and 'valid' can be changed for different train/valid set splitting
splits = {
    'human' : {
        'train': (1, 2, 3, 4, 5, 6, 7,           11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22),
        'valid': (8, ),
        'test' : (9, 10)
    },
    'mouse' : {
        'test': (1, 2, 3, 4),
        'plot': (4,),
        'all': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
    },
    'zebrafish':{
        'test': (1,2,3,4),
    }
}

human_total_length = 3088269832