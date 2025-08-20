#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=24GB
#PBS -l walltime=05:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training for TVC010 (SoyAgeing-R3) - PKB params from Table.md
python3 train.py --dataset soy_ageing_r3 --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --augmentation pkb --pkb-n 10 --pkb-a-frac 0.15 --pkb-sigma 2.1 --pkb-views 10 --pkb-placement random --color-jitter --hflip --rotate --save-best >> TVC012.log
