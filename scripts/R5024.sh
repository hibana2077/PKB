#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=60GB           
#PBS -l walltime=25:30:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# R5024: SoyGlobal, resnet50, PKB placement random, a-frac 0.25, sigma 2.0
python3 train.py --dataset soyglobal --model resnet50 --pretrained --augmentation pkb --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random --hflip --rotate --save-best >> R5024.log
