#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=32GB
#PBS -l walltime=40:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training with PKB models
python3 train.py --dataset cub_200_2011 --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --augmentation pkb --pkb-n 4 --pkb-a-frac 0.25 --pkb-sigma 1.5 --pkb-views 8 --pkb-placement random --color-jitter --hflip --rotate >> TVD001.log
python3 train.py --dataset cub_200_2011 --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --augmentation pkb --pkb-n 4 --pkb-a-frac 0.30 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random --color-jitter --hflip --rotate >> TVD002.log
python3 train.py --dataset cub_200_2011 --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --augmentation pkb --pkb-n 6 --pkb-a-frac 0.20 --pkb-sigma 1.5 --pkb-views 8 --pkb-placement random --color-jitter --hflip --rotate >> TVD003.log
python3 train.py --dataset cub_200_2011 --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --augmentation pkb --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random --color-jitter --hflip --rotate >> TVD004.log
python3 train.py --dataset cub_200_2011 --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --augmentation pkb --pkb-n 6 --pkb-a-frac 0.30 --pkb-sigma 2.5 --pkb-views 8 --pkb-placement random --color-jitter --hflip --rotate >> TVD005.log
python3 train.py --dataset cub_200_2011 --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --augmentation pkb --pkb-n 8 --pkb-a-frac 0.25 --pkb-sigma 1.5 --pkb-views 8 --pkb-placement random --color-jitter --hflip --rotate >> TVD006.log
python3 train.py --dataset cub_200_2011 --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --augmentation pkb --pkb-n 8 --pkb-a-frac 0.30 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random --color-jitter --hflip --rotate >> TVD007.log
python3 train.py --dataset cub_200_2011 --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --augmentation pkb --pkb-n 8 --pkb-a-frac 0.30 --pkb-sigma 2.0 --pkb-views 2 --pkb-placement random --color-jitter --hflip --rotate >> TVD008.log