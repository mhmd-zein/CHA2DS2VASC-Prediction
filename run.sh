#!/bin/bash
#$ -N rasta_train
#$ -l gpu=1
#$ -l h_rt=24:00:00
#$ -o logs/$JOB_ID.out
#$ -e logs/$JOB_ID.err
#$ -cwd

source $(conda info --base)/etc/profile.d/conda.sh
conda activate rasta_env

mkdir -p logs


python run_training.py \
     --data_dir Dataset/RASTA2 \
     --epochs 80 \
     --Name_Exp 5_fold_sup \
     --k_fold 5 \
     --dropout 0.4 \
     --input_channels 3 \
     --needed_modalities sup \

python run_training.py \
     --data_dir Dataset/RASTA2 \
     --epochs 80 \
     --Name_Exp 5_fold_deep \
     --k_fold 5 \
     --dropout 0.4 \
     --input_channels 3 \
     --needed_modalities deep \

python run_training.py \
     --data_dir Dataset/RASTA2 \
     --epochs 80 \
     --Name_Exp 5_fold_perf_map \
     --k_fold 5 \
     --dropout 0.4 \
     --input_channels 3 \
     --needed_modalities perf_maps \