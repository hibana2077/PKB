#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
#PBS -l mem=16GB           
#PBS -l walltime=02:00:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training with PKB models
python3 train.py --dataset cotton80 --model resnet50 --pretrained --augmentation pkb --pkb-n 12 --pkb-a-frac 0.20 --pkb-sigma 3.0 --pkb-views 12 --pkb-placement random --hflip --rotate --save-best >> J032.log
