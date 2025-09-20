#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=64GB
#PBS -l walltime=00:15:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ../..
python3 vis.py --compare-two-models --checkpoint-base "outputs/tvit_ct80_base.pth" --checkpoint-pkb  "outputs/tvit_ct80_pkb.pth" --model-base tiny_vit_21m_384 --model-pkb tiny_vit_21m_384 --dataset cub_200_2011 --do-tsne --do-umap --do-pca --top-k 15 --marker-size 24 --out-dir "./outputs/vis2"