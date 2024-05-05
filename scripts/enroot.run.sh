#!/usr/bin/env bash

#SBATCH --constraint=dgx
#SBATCH --exclude=dgx-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gpus=2
#SBATCH --time=0-00:30:00
#SBATCH --partition=short
#SBATCH --export=ALL,HYDRA_FULL_ERROR=1
#SBATCH --account=mandziuk-lab
#SBATCH --output=logs/slurm-%j.log

date "+%Y-%m-%d %H:%M:%S"
echo "SLURMD_NODENAME: ${SLURMD_NODENAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Enroot version: $(enroot version)"
echo "Command: python ${1}" "${@:2}"

enroot start \
    --rw \
    -c ~/Universal_AVR_Model/scripts/enroot_config.sh \
    -m ~/Universal_AVR_Model/src:/app/src:ro:x-create=dir,bind \
    -m ~/Universal_AVR_Model/.env:/app/.env:ro:x-create=file,bind \
    -m /mnt/evafs/groups/mandziuk-lab/akaminski/datasets:/app/data:ro:x-create=dir,bind \
    -m /etc/slurm:/etc/slurm \
    universal-avr-system-latest \
    "${1}" "${@:2}"
#    -m /mnt/evafs/groups/mandziuk-lab/mmalkinski/datasets/PGM:/app/data2/pgm \
echo "Enroot exited with code: $?"
date "+%Y-%m-%d %H:%M:%S"
