#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=20GB           
#PBS -l walltime=24:30:00  
#PBS -l wd                 
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

nvidia-smi >> gpu-info-v100.txt
source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training (CUB-200, Resnet50, PKB augmentation)
python3 train.py --dataset cub_200_2011 --model resnet50 --pretrained --augmentation pkb --pkb-n 20 --pkb-a-frac 0.085 --pkb-sigma 1.2 --pkb-views 7 --pkb-placement contiguous --hflip --rotate --save-best >> P08.log