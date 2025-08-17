#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
#PBS -l mem=32GB           
#PBS -l walltime=01:00:22  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training with PKB models (Soybean - TVB004)
python3 train.py --dataset soybean --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --augmentation pkb --pkb-n 6 --pkb-a-frac 0.30 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random --color-jitter --hflip --rotate --save-best >> TVB004.log
