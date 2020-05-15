#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=128GB
#SBATCH --time=72:00:00
#SBATCH --job-name=Inpainting
#SBATCH --mail-type=END
#SBATCH --mail-user=alexdong@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
source /home/awd275/.bashrc
conda activate Inpainting

cd /home/awd275/context_encoder_pytorch/reports

echo "starting python run"
/home/awd275/miniconda3/envs/Inpainting/bin/python make_PCA_on_celeb_orig.py &>log_pca_run1.txt
#/home/awd275/miniconda3/envs/Inpainting/bin/python make_PCA_on_celeb_recons.py &>log_pca_run2.txt
#/home/awd275/miniconda3/envs/Inpainting/bin/python make_PCA_on_celeb_orig_cropped.py &>log_pca_run3.txt
#/home/awd275/miniconda3/envs/Inpainting/bin/python make_PCA_on_celeb_recons_cropped.py &>log_pca_run4.txt


exit

