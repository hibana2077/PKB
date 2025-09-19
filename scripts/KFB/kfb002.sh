#!/bin/bash
#PBS -P cp23
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=20GB           
#PBS -l walltime=24:30:00  
#PBS -l wd                  
#PBS -l storage=scratch/cp23

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/cp23/lw4988/PKB/.venv/bin/activate

cd ../..

# Single run generated from grid (no --color-jitter)
python3 train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random >> kfb002.log 2>&1
