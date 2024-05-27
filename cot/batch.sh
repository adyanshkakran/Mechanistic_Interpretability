#!/bin/bash
#SBATCH --ntasks-per-node 10
#SBATCH -w gnode047
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output2.txt
#SBATCH --time=12:00:00

source /home2/adyansh/miniconda3/bin/activate

conda activate /home2/adyansh/dinner_pool/rsai/

python3 /home2/adyansh/dinner_pool/Mechanistic_Interpretability/cot/inference.py
