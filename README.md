# Evo2HiC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17917911.svg)](https://doi.org/10.5281/zenodo.17917911)

Evo2HiC is a multimodal and lightweight foundation model  for jointly modeling genomic sequences and structures.

## Introduction
<details>
<summary>Evo2HiC is a multimodal and lightweight foundation model  for jointly modeling genomic sequences and structures.</summary>

Understanding how genomic sequences shape three-dimensional (3D) genome architecture is fundamental to interpreting diverse biological processes. Although previous studies have demonstrated that sequence information can predict chromatin organization, they are unable to capture cell-type specific structures due to the lack of consideration of Hi-C datasets, which are widely available and capture rich structural information across biosamples.
Recently, DNA foundation models have demonstrated encouraging performance in capturing long-range genomic dependencies, holding promise for modeling chromatin interactions. However, the extremely high computational cost of running these models limits their applicability to Hi-C analysis, which requires genome-wide sequence embeddings.
Here, we present Evo2HiC, a lightweight foundation model for jointly modeling genomic sequences and structures. The key idea of Evo2HiC is to distill a large-scale DNA foundation model, Evo~2 (7B), into a compact encoder, while guiding the distillation with Hi-C data to preserve genomic features critical for 3D genome analysis.
When predicting Hi-C contact matrix using genome sequence, Evo2HiC improved Spearman correlation by 10.9\% over Orca.
Moreover, Evo2HiC achieved the best overall Pearson correlation when predicting five representative epigenomic assays by jointly embedding Hi-C and sequence information. Interpretation analysis of Evo2HiC revealed its ability to identify cell type–specific sequence motifs that explain changes in epigenomic signals. Finally, we demonstrated the cross-species generalizability of Evo2HiC on 177 species from the DNA Zoo dataset for Hi-C resolution enhancement.
In summary, Evo2HiC is a multimodal foundation model that integrates genome sequence and 3D chromatin structure information, substantially reduces computational cost while maintaining state-of-the-art accuracy across multiple chromatin analysis tasks, enables the identification of cell type-specific motifs, and demonstrates robust generalizability across species.

</details>

## Prerequisites

1. Installing miniconda with following commands:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```
For more details about miniconda, check [the official document](https://www.anaconda.com/docs/getting-started/miniconda/main).

2. Install and activate required packages:
```bash
conda env create -f Evo2HiC.yaml
conda activate Evo2HiC
```

3. add current path to python path:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Pretrained Models

Pretrained Evo2HiC models are available on Zenodo:
https://doi.org/10.5281/zenodo.17917911

`pretrained_weights` contains the pretrained encoders only and must be finetuned before predicting contact maps.
`resolution_enhancement`, `epi_prediction`, `seq2hic_{H1ESC,HFF}` and `seq2hic_cg_{H1ESC,HFF}` are finetuned models.

Files referenced in `config.py` but not included in this repository: [juicer_tools v2.20.00](https://github.com/aidenlab/Juicebox/releases/tag/v2.20.00),
the [Umap k100 mappability track](https://hgdownload.soe.ucsc.edu/gbdb/hg38/hoffmanMappability/k100.Umap.MultiTrackMappability.bw),
a MEME-format motif database (e.g. JASPAR) and [3d-dna](https://github.com/aidenlab/3d-dna).

## Usage
### Pretraining

1. Collect hic data from ENCODE and 4DN (full index in 'data/hic_index.tsv'). Save the hic data to hic_data_dir specified in config.py.

2. Run this command to start pretraining:

```bash
accelerate launch train/pretrain.py
```

The training process will be recorded by weight-and-bias(wandb).
Read 'train/pretrain.py' if you want to change hyper-parameters.

### inference based on pretrained checkpoint

1. Calculate 1D DNA embeddings:

```bash
python inference/inference_dna_embeds.py -ckpt your_pretrained_ckpt
```

2. Calculate 2D DNA embeddings and Hi-C embeddings for retrieval:

```bash
python inference/retrieval_siglip.py -ckpt your_pretrained_ckpt
```

### Finetune pretrained checkpoint for predicting Hi-C contact matrix using genome sequences

1. Finetine the pretrained model:

```bash
accelerate launch train/train_CDNA2d_Seq2HiC.py --initialize your_pretrained_ckpt
```

The Seq2HiC input depends on the sequencing depth of the input `.hic` file; for libraries much shallower or deeper
than the training data, match the input's mean and standard deviation to the training data.

### Finetune pretrained checkpoint for predicting Hi-C in Orca's setting (Evo2HiC-cg)

1. Prepare the coarse-grained Micro-C target and the auxiliary peak targets from Orca's resources:

```bash
python scripts/make_cg_targets.py your_mcool your_hic_accession --scale your_scale
python scripts/make_aux_peaks.py your_peaks.bed.sorted.gz your_peaks.bed.sorted.features your_hic_accession
```

2. Finetune the pretrained model:

```bash
accelerate launch train/train_CDNA2d_Seq2HiC.py -target cg --target-specified your_hic_accession --aux-peaks your_hic_accession --initialize-dna-encoder your_pretrained_ckpt
```

The full hyper-parameters are in the `args.json` of the released checkpoints. Checkpoints were selected after
recomputing batch-normalization statistics (`--recalibrate-bn`, then `--valid-only`).

### Finetune pretrained checkpoint for Hi-C resolution enhancement

1. Finetine the pretrained model:

```bash
accelerate launch train/train_CDNA2d_ResEnh.py --initialize your_pretrained_ckpt
```

### Finetune pretrained checkpoint for predicting epigenomic profiles

1. Finetine the pretrained model:

```bash
accelerate launch train/train_CDNA1d.py --initialize your_pretrained_ckpt
```

### Inference finetuned models

1. Predicting Hi-C maps with finetuned model (Seq2HiC or ResEnh):

```bash
python inference/inference_CDNA2d.py -ckpt your_pretrained_ckpt --options_about_data
```

2. Predicting Epigenomic profiles with finetuned model:

```bash
python inference/inference_CDNA1d.py -ckpt your_pretrained_ckpt --options_about_data
```

### Model interpretation (motif discovery)

Evo2HiC can be interpreted to reveal the sequence motifs that drive its predictions.
`interpretation/inter_CDNA1d.py` attributes a finetuned epigenomic-prediction model (CDNA1d, which
predicts epigenomic profiles from DNA + Hi-C) back to the input nucleotides, then calls
[TF-MoDISco](https://github.com/jmschrei/tfmodisco-lite) to discover motifs and match them against a
motif database.

This needs a few extra packages that are not in `Evo2HiC.yaml`:

```bash
pip install captum tangermeme modisco-lite
```

Compute per-base attributions and run motif discovery for each cell type / epigenomic assay:

```bash
python interpretation/inter_CDNA1d.py -ckpt your_epi_ckpt --save-dir human_inter -m deepliftshap
```

`-m` selects the attribution method (`deepliftshap`, `IG`, or `perturb`). Results are written under
`<checkpoint_dir>/human_inter/<cellline>/<assay>/`, including the sampled sequences
(`seq_sample.npy`), attributions (`attr_sample.npy`), the TF-MoDISco results (`modisco_results.h5`),
and an HTML `report/` matching discovered motifs against `motif_data` (`data/dna/motifs.meme` in
`config.py`).

## Tutorial: Training on a new dataset

This is a step-by-step walkthrough of how to run Evo2HiC on your own data — a new biosample or a
new species. It expands on the [Usage](#usage) section above with the exact files you need to
prepare and the config entries you need to edit. All paths and flags refer to the scripts in this
repository; defaults are defined in `config.py` and `dataset/data_arguments.py`.

Evo2HiC is trained jointly on genome sequence, Evo~2 embeddings, and Hi-C contact maps. To add
your own dataset you provide three ingredients — a reference genome (FASTA), precomputed Evo~2
embeddings, and one or more Hi-C maps — then register them in `config.py` and run training.

### Step 1 — Add the reference genome
Place the genome FASTA on disk and register it in `config.py`:

```python
# config.py
references = {'human': 'hg38', 'mouse': 'mm10', 'mynewspecies': 'myasm1'}
DNA_map    = {..., 'mynewspecies': 'data/dna/mynewspecies/myasm1.fa'}
# Mappability is optional; set to None if you do not have a Umap track.
mappability_map = {..., 'mynewspecies': None}
```

### Step 2 — Precompute Evo~2 embeddings
Evo2HiC distills Evo~2 (7B); the sequence embeddings are generated once per chromosome and
cached as memory-mapped arrays. Run, for each chromosome:

```bash
python misc/generate_evo2_embedding.py \
    -c <chromosome_number> \
    -r myasm1 \
    --DNA-file-path data/dna/mynewspecies/myasm1.fa
```

This writes embeddings to `data/dna/mynewspecies/myasm1_evo2_7b/`. Point `config.py` at it:

```python
evo2_embedding_map = {..., 'mynewspecies': 'data/dna/mynewspecies/myasm1_evo2_7b/'}
```

> Note: this step requires the [`evo2`](https://github.com/ArcInstitute/evo2) package and a GPU.
> If you only need the lightweight distilled encoder for inference, you can skip Evo~2 embeddings
> at training/inference time by leaving `--evo2-option No` (the default).

### Step 3 — Register the Hi-C data
Save your `.hic` / `.cool` files under the directory given by `hic_data_dir` in `config.py`, and
add one row per dataset to the index TSV (`data/hic_index.tsv`), following the existing column
layout. The `split` column (`train`/`valid`/`test`) and the per-chromosome split in `config.py`'s
`splits` dictionary control how data is partitioned:

```python
# config.py
splits = {
    ...,
    'mynewspecies': {
        'train': (1, 2, 3, 4, 5, 6, 7),
        'valid': (8,),
        'test':  (9, 10),
    },
}
```

### Step 4 — Pretrain (optional) or finetune
To pretrain the joint sequence–structure encoder from scratch on the new data:

```bash
accelerate launch train/pretrain.py
```

In most cases it is faster and more accurate to finetune the released checkpoint
(see [Pretrained Models](#pretrained-models)) for the task you care about, e.g. predicting Hi-C
from sequence on the new biosample:

```bash
accelerate launch train/train_CDNA2d_Seq2HiC.py --initialize checkpoints/pretrained_weights/model.pt
```

Pass `-sp mynewspecies` and the relevant data flags (resolution, normalization, etc.) defined in
`dataset/data_arguments.py`. Training is logged to Weights & Biases (wandb).

The same `--initialize` pattern applies to the other tasks: `train/train_CDNA2d_ResEnh.py` for
Hi-C resolution enhancement and `train/train_CDNA1d.py` for epigenomic profile prediction.

### Step 5 — Run inference with your trained model
Each finetuned checkpoint is saved together with an `args.json` recording its training
configuration, which the inference scripts read automatically. Point the inference script at your
checkpoint and select the data with the same `-sp`/`--split` flags used during training.

Predict Hi-C contact maps (Seq2HiC or resolution enhancement):

```bash
python inference/inference_CDNA2d.py -ckpt your_finetuned_ckpt \
    -sp mynewspecies --split test --save-dir my_results
```

Predict epigenomic profiles:

```bash
python inference/inference_CDNA1d.py -ckpt your_finetuned_ckpt \
    --split test --save-dir my_results
```

Results are written under a subdirectory of the checkpoint named after `--save-dir`. See
`dataset/data_arguments.py` for the full list of data options (resolution, downsample rate,
normalization, etc.).