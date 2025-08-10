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
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training with PKB models
python3 train.py --dataset cotton80 --model resnet50 --pretrained --augmentation pkb --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 20 --pkb-placement random --color-jitter --hflip --rotate --save-best >> J002.log
