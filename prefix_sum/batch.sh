#!/bin/bash
#SBATCH --ntasks-per-node 10
#SBATCH -w gnode092
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output.txt
#SBATCH --time=16:00:00

source /home2/adyansh/miniconda3/bin/activate

conda activate /home2/adyansh/dinner_pool/rsai/

python3 /home2/adyansh/dinner_pool/Mechanistic_Interpretability/prefix_sum/train.py
python3 /home2/adyansh/dinner_pool/Mechanistic_Interpretability/prefix_sum/inference.py
