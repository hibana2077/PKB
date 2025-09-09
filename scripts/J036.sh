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
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training (Stanford Cars, Resnet50, PKB augmentation)
python3 train.py --dataset stanford_cars --model resnet50 --pretrained --augmentation pkb --pkb-n 12 --pkb-a-frac 0.06 --pkb-sigma 0.6 --pkb-views 8 --pkb-placement random --hflip --rotate --save-best >> J036.log
