#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=8GB
#PBS -l walltime=00:05:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ../..
python3 vis.py --checkpoint ./outputs/best_resnet50_stanford_cars.pth --dataset stanford_cars --model resnet50 --split test --do-gradcam --gradcam-samples 8 --device cuda --out-dir ./outputs/vis_gradcam_stanford_cars_tsne_r50_test