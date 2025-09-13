#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
#PBS -l mem=32GB           
#PBS -l walltime=02:30:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# R3419: SoyAgeing-R6, resnet34.a1_in1k, PKB placement contiguous, a-frac 0.25, sigma 2.0
python3 train.py --dataset soy_ageing_r6 --model resnet34.a1_in1k --pretrained --augmentation pkb --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous --hflip --rotate --save-best >> R3419.log
