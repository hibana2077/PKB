#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=2           
#PBS -l ncpus=24           
#PBS -l mem=40GB           
#PBS -l walltime=01:05:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training with PKB models
python3 train.py --dataset soybean --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --augmentation pkb --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 16 --pkb-placement random --color-jitter --hflip --rotate --save-best --multi-gpu >> EXA002.log