#!/bin/bash
# LWT

#SBATCH --job-name=MMCls_VentALL_CXR
#SBATCH --account=kayvan99
#SBATCH --partition=spgpu
#SBATCH --time=4-23:59:59
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=./%x-%j
#SBATCH --error=./%x-%j
#SBATCH --mail-user=liuwent@med.umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

if [[ $SLURM_JOB_NODELIST ]] ; then
   echo "Running on"
   scontrol show hostnames $SLURM_JOB_NODELIST
fi

module load gcc/11.2.0
module load cuda
module load cudnn
module load python3.10-anaconda/2023.03

source ~/.bashrc
conda activate mamba

python -u 05b-Multi_Modalities_Models_Classification.py