#!/bin/bash
set +e
# Installation and download IPA dictionaries

#SBATCH -c 12
#SBATCH --mem=12GB
#SBATCH -G 1
#SBATCH -p gpu-preempt
#SBATCH --time 01:00:00
#SBATCH -o scripts/out/generate_images_%j.out
#SBATCH --mail-type END

module load miniconda/22.11.1-1
conda activate sinr_icml

python /home/oyilmazel_umass_edu/inaturalist-sinr/viz_map.py  --name xsmall --model_path /home/oyilmazel_umass_edu/inaturalist-sinr/fine-tuned/demo/fine_tune_xsmall_lr.pt --taxa_id 5165
python /home/oyilmazel_umass_edu/inaturalist-sinr/viz_map.py  --name small --model_path /home/oyilmazel_umass_edu/inaturalist-sinr/fine-tuned/demo/fine_tune_small_lr.pt --taxa_id 5165
python /home/oyilmazel_umass_edu/inaturalist-sinr/viz_map.py --name vanilla --model_path /home/oyilmazel_umass_edu/inaturalist-sinr/pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt --taxa_id 5165

python /home/oyilmazel_umass_edu/inaturalist-sinr/viz_map.py  --name xsmall --model_path /home/oyilmazel_umass_edu/inaturalist-sinr/fine-tuned/demo/fine_tune_xsmall_lr.pt --taxa_id 52173
python /home/oyilmazel_umass_edu/inaturalist-sinr/viz_map.py --name small --model_path /home/oyilmazel_umass_edu/inaturalist-sinr/fine-tuned/demo/fine_tune_small_lr.pt --taxa_id 52173
python /home/oyilmazel_umass_edu/inaturalist-sinr/viz_map.py --name vanilla --model_path /home/oyilmazel_umass_edu/inaturalist-sinr/pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt --taxa_id 52173

python /home/oyilmazel_umass_edu/inaturalist-sinr/viz_map.py  --name xsmall --model_path /home/oyilmazel_umass_edu/inaturalist-sinr/fine-tuned/demo/fine_tune_xsmall_lr.pt
python /home/oyilmazel_umass_edu/inaturalist-sinr/viz_map.py --name small --model_path /home/oyilmazel_umass_edu/inaturalist-sinr/fine-tuned/demo/fine_tune_small_lr.pt
python /home/oyilmazel_umass_edu/inaturalist-sinr/viz_map.py --name vanilla --model_path /home/oyilmazel_umass_edu/inaturalist-sinr/pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt
