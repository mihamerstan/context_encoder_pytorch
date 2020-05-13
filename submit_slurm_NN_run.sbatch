#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128GB
#SBATCH --time=72:00:00
#SBATCH --job-name=ContextEncoder
#SBATCH --mail-type=END
#SBATCH --mail-user=alexdong@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
source /home/awd275/.bashrc
conda activate Inpainting

cd /home/awd275/context_encoder_pytorch

echo "starting python run"
current_date_time="`date +%H%M%S`";
echo $current_date_time;
/home/awd275/miniconda3/envs/Inpainting/bin/python Run_model_on_images.py
echo $current_date_time;
exit
