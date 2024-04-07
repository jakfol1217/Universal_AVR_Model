#!/bin/bash

TARGET_DIR=${1:-"."}  # The directory where the data will be downloaded

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/raw
mkdir -p $TARGET_DIR/bongard_hoi

URL=https://zenodo.org/record/7079175/files/bongard_hoi_images.tar
URL_LABELS=https://zenodo.org/record/7079175/files/bongard_hoi_annotations.tar

echo "--------------------------------------------------------------------------------"
echo "Downloading BongardHOI dataset into $TARGET_DIR/bongard_hoi"
echo -e "--------------------------------------------------------------------------------\n"

wget -c $URL -O $TARGET_DIR/raw/bongard_hoi_images.tar
tar -xf $TARGET_DIR/raw/bongard_hoi_images.tar $TARGET_DIR/bongard_hoi

echo "--------------------------------------------------------------------------------"
echo "Downloading BongardHOI dataset annotations into $TARGET_DIR/bongard_hoi"
echo -e "--------------------------------------------------------------------------------\n"


wget -c $URL_LABELS -O $TARGET_DIR/raw/bongard_hoi_annotations.tar
tar -xf $TARGET_DIR/raw/bongard_hoi_annotations.tar $TARGET_DIR/bongard_hoi
