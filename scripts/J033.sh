#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
#PBS -l mem=32GB           
#PBS -l walltime=02:00:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training with PKB models
python3 train.py --dataset cotton80 --model resnet50 --pretrained --augmentation pkb --pkb-n 14 --pkb-a-frac 0.30 --pkb-sigma 2.5 --pkb-views 10 --pkb-placement random --hflip --rotate --save-best >> J033.log
