#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=20GB           
#PBS -l walltime=00:55:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training with PKB models
python3 train.py --dataset cotton80 --model resnet50 --pretrained --augmentation pkb --pkb-n 8 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 4 --pkb-placement random --color-jitter --hflip --rotate --save-best >> J002.log
