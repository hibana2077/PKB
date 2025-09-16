#!/bin/bash
#PBS -P cp23
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=60GB           
#PBS -l walltime=25:30:00  
#PBS -l wd                  
#PBS -l storage=scratch/cp23

module load cuda/12.6.2

source /scratch/cp23/lw4988/PKB/.venv/bin/activate

cd ..
# R5030: NAbird, resnet50, Base augmentation (no PKB)
python3 train.py --dataset nabirds --model resnet50 --pretrained --hflip --rotate --save-best >> R5030_cp23.log
