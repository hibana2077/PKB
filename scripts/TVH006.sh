#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
#PBS -l mem=32GB           
#PBS -l walltime=24:00:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training (TVH006)
python3 train.py --dataset cub_200_2011 --model tiny_vit_21m_384 --pretrained --hflip --rotate --save-best --augmentation pkb --pkb-n 10 --pkb-a-frac 0.15 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement contiguous >> TVH006.log
