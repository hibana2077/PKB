#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
#PBS -l mem=60GB           
#PBS -l walltime=24:00:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training (NAbirds - TVH006)
python3 train.py --dataset nabirds --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --augmentation pkb --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 5.0 --pkb-views 10 --pkb-placement contiguous --hflip --rotate --save-best >> TVH006.log
