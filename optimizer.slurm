#!/bin/bash

#############################

# Your job name (displayed by the queue)
#SBATCH -J reservoir_optimizer

# Specify the number of desired jobs in your job array (here 50)
#SBATCH --array=0-49

# Specify the number of nodes(nodes=) and the number of cores per nodes(tasks-pernode=) to be used
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=15

# change working directory
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

#############################

# run optimization script from virtual environment
venv/bin/python optimization.py \
    --trials 5 \
    --study_name classifier_optimizer \
    --use_oscillators=True \
    --nodes 50 \
    --binary=False \
    --instances=1000 \
    --job_id=${SLURM_ARRAY_TASK_ID}

# all done
echo "Optimization Finished"
