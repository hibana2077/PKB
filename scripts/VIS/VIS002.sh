#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=8GB
#PBS -l walltime=00:05:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ../..
python3 vis.py --checkpoint ./outputs/best_tiny_vit_21m_384.dist_in22k_ft_in1k_soybean.pth --dataset soybean --split test --model tiny_vit_21m_384.dist_in22k_ft_in1k --do-tsne --first-n-classes 10 --out-dir ./outputs/vis_soybean_tsne_tv_test