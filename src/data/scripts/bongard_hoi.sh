#!/bin/bash

TARGET_DIR=${1:-"."}  # The directory where the data will be downloaded

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/bongard_hoi

URL=https://zenodo.org/record/7079175/files/bongard_hoi_images.tar
URL_LABELS=https://zenodo.org/record/7079175/files/bongard_hoi_annotations.tar

echo "--------------------------------------------------------------------------------"
echo "Downloading BongardHOI dataset into $TARGET_DIR/bongard_hoi"
echo -e "--------------------------------------------------------------------------------\n"

wget -c $URL -O - | tar -x -C $TARGET_DIR/bongard_hoi
