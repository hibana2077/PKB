#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=20GB           
#PBS -l walltime=00:59:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training with PKB models
python3 train.py --dataset soy_ageing_r5 --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --color-jitter --hflip --rotate --save-best >> A006.log