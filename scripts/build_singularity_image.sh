#!/bin/bash

#SBATCH --constraint=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=0-2:00:00
#SBATCH --export=ALL
#SBATCH --account=mandziuk-lab

set -ex

timestamp="$(date +'%Y-%m-%d_%H-%M-%S')"
DOCKER_BUILD_DIR='/vagrant'
DOCKER_FILE_PATH='/vagrant/docker/nvidia.Dockerfile'
DOCKER_IMAGE_URI='kaminskia/universal-avr-system-nvidia:latest'
OUTPUT_DIR_HOST='/raid/shared/akaminski'
OUTPUT_DIR_GUEST='/output'
OUTPUT_TMP_HOST='/mnt/evafs/groups/mandziuk-lab/akaminski/singularity/tmp'
OUTPUT_FILENAME="universal-avr-system-nvidia_${timestamp}.tar.gz"
SINGULARITY_CONTAINER_PATH="/mnt/evafs/groups/mandziuk-lab/akaminski/singularity/universal-avr-system-nvidia_${timestamp}.sif"

env \
  VAGRANT_EXPERIMENTAL=1 \
  DOCKER_BUILD_DIR="${DOCKER_BUILD_DIR}" \
  DOCKER_FILE_PATH="${DOCKER_FILE_PATH}" \
  DOCKER_IMAGE_URI="${DOCKER_IMAGE_URI}" \
  OUTPUT_DIR_HOST="${OUTPUT_DIR_HOST}" \
  OUTPUT_DIR_GUEST="${OUTPUT_DIR_GUEST}" \
  OUTPUT_TMP_HOST="${OUTPUT_TMP_HOST}" \
  OUTPUT_FILENAME="${OUTPUT_FILENAME}" \
  vagrant up --provision
echo "Provisioned virtual machine and saved docker image to: ${OUTPUT_DIR_HOST}/${OUTPUT_FILENAME}"

vagrant suspend
echo "Suspended virtual machine"

singularity build "${SINGULARITY_CONTAINER_PATH}" "docker-archive://${OUTPUT_DIR_HOST}/${OUTPUT_FILENAME}"
if [ $? -ne 0 ]; then
  echo "Failed to create singularity container ${SINGULARITY_CONTAINER_PATH}"
  exit 1
fi
mkdir -p ~/singularity
ln -sf "${SINGULARITY_CONTAINER_PATH}" ~/singularity/universal-avr-system-nvidia-latest.sif
echo "Created new singularity container ${SINGULARITY_CONTAINER_PATH} from ${OUTPUT_DIR_HOST}/${OUTPUT_FILENAME}"
