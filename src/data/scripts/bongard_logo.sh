#!/bin/bash

TARGET_DIR=${1:-"."}  # The directory where the data will be downloaded

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/bongard_logo

URL=https://drive.google.com/file/d/1-1j7EBriRpxI-xIVqE6UEXt-SzoWvwLx/view
ID=1-1j7EBriRpxI-xIVqE6UEXt-SzoWvwLx

echo "--------------------------------------------------------------------------------"
echo "Downloading BongardLOGO dataset into $TARGET_DIR/bongard_logo"
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

gdown $ID -c -O $TARGET_DIR/temp_bongard_logo.zip

unzip -q $TARGET_DIR/temp_bongard_logo.zip -d $TARGET_DIR/bongard_logo

if succefully unzipped remove temp file
if [ $? -eq 0 ]; then
    rm -rf $TARGET_DIR/bongard_logo/__MACOSX
    rm $TARGET_DIR/temp_bongard_logo.zip
fi
