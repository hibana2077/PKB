#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=64GB           
#PBS -l walltime=09:00:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training with PKB models
python3 train.py --dataset soygene --model efficientnet_b0.ra4_e3600_r224_in1k --pretrained --color-jitter --hflip --rotate --train-crop 224 --save-best >> E003.log