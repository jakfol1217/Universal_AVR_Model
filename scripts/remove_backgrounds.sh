#!/usr/bin/env bash

#SBATCH --constraint=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16GB
#SBATCH --time=0-12:00:00
#SBATCH --partition=short 
#SBATCH --export=ALL,HYDRA_FULL_ERROR=1
#SBATCH --account=mandziuk-lab

date "+%Y-%m-%d %H:%M:%S"
echo "SLURMD_NODENAME: ${SLURMD_NODENAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "singularity version: $(singularity version)"
echo "nvidia-container-toolkit version: $(nvidia-container-toolkit -version)"
echo "nvidia-container-cli info: $(nvidia-container-cli info)"
echo "Command: python ${1}" "${@:2}"

echo "$(date '+%Y-%m-%d %H:%M:%S'): [remove_backgrounds.sh], JOB_ID ${SLURM_JOB_ID}, python ${1}" "${@:2}" >> /mnt/evafs/groups/mandziuk-lab/akaminski/out/commands.log

singularity run \
    --nv \
    --bind ~/Universal_AVR_Model/remove_backgrounds.py:/app/remove_backgrounds.py:ro \
    --bind ~/Universal_AVR_Model/src:/app/src:ro \
    --bind ~/Universal_AVR_Model/.env:/app/.env:ro \
    --bind /mnt/evafs/groups/mandziuk-lab/akaminski/datasets:/app/data:rw \
    --bind /mnt/evafs/groups/mandziuk-lab/akaminski/out:/app/out:rw \
    ~/singularity/universal-avr-system-nvidia-latest.sif \
    "${1}" "${@:2}"
echo "Singularity exited with code: $?"
date "+%Y-%m-%d %H:%M:%S"