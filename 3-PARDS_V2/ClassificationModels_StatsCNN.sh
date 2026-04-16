#!/bin/bash
# LWT

#SBATCH --job-name=ClassificationModels(StatsCNN)   ## Name of the job for the scheduler
#SBATCH --account=kayvan0                           ## name of the resource account (who is paying for the compute time)
#SBATCH --partition=spgpu                           ## name of the queue to submit the job to.   
                                                    ##(Choose from: standard, debug, largemem, gpu   ---  accordingly)
#SBATCH --time=9-11:59:59                           ## Maximum length of time you are reserving the resources for 
                                                    ## (if job ends sooner, bill is based on time used)
#SBATCH --nodes=1                                   ## number of nodes you are requesting
#SBATCH --tasks-per-node=1                          ## how many instances of a running program do you want to have
#SBATCH --cpus-per-task=8                           ## how many cores do you want to reserve

#SBATCH --mem=128G                                  ## Memory requested for this job
#SBATCH --gres=gpu:1
#SBATCH --output=./%x-%j                            ## send output and error information to the file listed (optional: different name format than default)
#SBATCH --error=./%x-%j
#SBATCH --mail-user=liuwent@med.umich.edu           ## send email notifications to umich email listed
#SBATCH --mail-type=BEGIN,END                       ## when to send email (standard values are: 
                                                    ## NONE, BEGIN, END, FAIL, REQUEUE, ALL.  See documentation for others)

# I recommend using the following lines so that some output is put in your output file as an indicator your script is working

if [[ $SLURM_JOB_NODELIST ]] ; then
   echo "Running on"
   scontrol show hostnames $SLURM_JOB_NODELIST
fi

# Load modules
module load gcc/11.2.0
module load cuda
module load cudnn
module load python3.10-anaconda/2023.03

# Activate your env
source ~/.bashrc
conda activate mamba

#  Put your job commands after this line run this command before submitting the script:
python -u ClassificationModels_StatsCNN.py