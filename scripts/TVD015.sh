#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=12GB
#PBS -l walltime=25:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training with PKB models
python3 train.py --dataset cub_200_2011 --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --augmentation pkb --pkb-n 7 --pkb-a-frac 0.14 --pkb-sigma 2.0 --pkb-views 2 --pkb-placement contiguous --hflip --rotate --save-best --output ./outputs/tvd015 >> TVD015.log