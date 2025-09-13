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
# R3411: SoyAgeing-R3, resnet34.tv_in1k, PKB placement dispersed, a-frac 0.25, sigma 2.0
python3 train.py --dataset soy_ageing_r3 --model resnet34.tv_in1k --pretrained --augmentation pkb --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed --hflip --rotate --save-best >> R3411.log
