#!/bin/bash

TARGET_DIR=${1:-"."}  # The directory where the data will be downloaded

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/dsprites

URL=https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz

echo "--------------------------------------------------------------------------------"
echo "Downloading dSprites dataset into $TARGET_DIR/dsprites"
echo -e "--------------------------------------------------------------------------------\n"

wget -c $URL -P $TARGET_DIR/dsprites
