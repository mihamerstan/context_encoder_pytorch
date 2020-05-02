#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64GB
#SBATCH --time=72:00:00
#SBATCH --job-name=MultiRF
#SBATCH --mail-type=END
#SBATCH --mail-user=alexdong@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
source /home/awd275/.bashrc
conda activate Inpainting

cd /home/awd275/context-encoder-pytorch/reports/

echo "starting python run"
/home/awd275/miniconda3/envs/Inpainting/bin/python run_PCA_on_celebfaces.py &>log_pca_run.txt
exit