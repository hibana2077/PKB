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
# R3433: NAbird, resnet34.tv_in1k, Base
python3 train.py --dataset nabird --model resnet34.tv_in1k --pretrained --hflip --rotate --save-best >> R3433.log