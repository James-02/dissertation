#!/bin/bash

#############################

# job name (displayed by the queue)
#SBATCH -J reservoir_optimizer

# number of desired jobs in job array
#SBATCH --array=0-49

# number of nodes
#SBATCH --nodes=1
# number of cores per nodes(tasks-pernode=) to be used
#SBATCH --ntasks-per-node=15

# Standard output is saved in this file
#SBATCH -o logs/slurm%A_%a.out

# Standard error messages are saved in this file
#SBATCH -e logs/slurm_%A_%a.err

# change working directory
#SBATCH --chdir=.

# send email updates
#SBATCH --mail-type=ALL

#############################

# useful information to print
echo "#############################"
echo "User:" $USER
echo "Date:" `date`
echo "Host:" `hostname`
echo "Directory:" `pwd`
echo "SLURM_JOBID:" $SLURM_JOBID
echo "SLURM_SUBMIT_DIR:" $SLURM_SUBMIT_DIR
echo "SLURM_JOB_NODELIST:" $SLURM_JOB_NODELIST
echo "#############################"

############################

module load Python/3.10
python3.10 -m venv venv
venv/bin/pip install -r requirements.txt

# run optimization script from virtual environment
venv/bin/python optimization.py \
    --trials 5\
    --study_name classifier_optimizer \
    --use_oscillators=True \
    --nodes 50 \
    --binary=False \
    --instances=1000 \
    --job_id=${SLURM_ARRAY_TASK_ID}

# all done
echo "Optimization Finished"