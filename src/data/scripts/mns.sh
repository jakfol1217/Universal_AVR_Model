#!/bin/bash

TARGET_DIR=${1:-"."}  # The directory where the data will be downloaded

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/mns

URL=https://drive.google.com/file/d/17KuL8KOIDAeRL-lD418oiDEm8bE6TEFb/view
ID=17KuL8KOIDAeRL-lD418oiDEm8bE6TEFb

echo "--------------------------------------------------------------------------------"
echo "Downloading MNS dataset into $TARGET_DIR/mns"
echo -e "--------------------------------------------------------------------------------\n"

# Check if unzip and gdown is installed if not install it

if ! [ -x "$(command -v unzip)" ]; then
    echo 'Error: unzip is not installed.' >&2
    exit 1
fi

if ! [ -x "$(command -v gdown)" ]; then
    echo 'Error: gdown is not installed.' >&2
    exit 1
fi

gdown $ID -c -O $TARGET_DIR/temp_mns.zip

unzip -q $TARGET_DIR/temp_mns.zip -d $TARGET_DIR/mns

# if succefully unzipped remove temp file
if [ $? -eq 0 ]; then
    rm $TARGET_DIR/temp_mns.zip
fi
