#!/bin/bash
# Installation and download IPA dictionaries

#SBATCH -c 12
#SBATCH --mem=12GB
#SBATCH --time 01:00:00
#SBATCH -o scripts/out/extract_data%j.out
#SBATCH --mail-type END

module load miniconda/22.11.1-1
conda activate sinr_icml

python /home/oyilmazel_umass_edu/inaturalist-sinr/data_extraction.py
