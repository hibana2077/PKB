#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=20GB           
#PBS -l walltime=25:30:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# R3429: Stanford_Cars, resnet34.a1_in1k, PKB placement dispersed, a-frac 0.25, sigma 2.0
python3 train.py --dataset stanford_cars --model resnet34.a1_in1k --pretrained --augmentation pkb --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed --hflip --rotate --save-best >> R3429.log
