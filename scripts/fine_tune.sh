#!/bin/bash
# Installation and download IPA dictionaries

#SBATCH -c 12
#SBATCH --mem=12GB
#SBATCH -G 1
#SBATCH -p gpu-preempt
#SBATCH --time 01:00:00
#SBATCH -o scripts/out/fine_tune_%j.out

module load miniconda/22.11.1-1
conda activate sinr_icml

python /home/oyilmazel_umass_edu/inaturalist-sinr/fine_tune_main.py
