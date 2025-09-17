#!/bin/bash
#PBS -P cp23
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=40GB
#PBS -l walltime=00:10:00  
#PBS -l wd                  
#PBS -l storage=scratch/cp23

module load cuda/12.6.2

source /scratch/cp23/lw4988/PKB/.venv/bin/activate

cd ..
# R3432: NAbird, resnet34.a1_in1k, PKB placement dispersed, a-frac 0.35, sigma 8.0
python3 train.py --dataset nabirds --model resnet34.a1_in1k --pretrained --augmentation pkb --pkb-n 10 --pkb-a-frac 0.35 --pkb-sigma 8.0 --pkb-views 2 --pkb-placement dispersed --hflip --rotate --save-best --epochs 10 >> R3432.log