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

source /scratch/cp23/sl5952/PKB/.venv/bin/activate

cd ../..

# Single run generated from grid (no --color-jitter)
python3 train.py --dataset stanford_cars --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.34 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous >> kfa141.log 2>&1
