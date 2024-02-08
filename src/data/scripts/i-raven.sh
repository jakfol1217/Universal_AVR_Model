#!/bin/bash

TARGET_DIR=${1:-"."}  # The directory where the data will be downloaded

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/i-raven

URL=https://drive.google.com/file/d/1SxhImd29PLtlvqXAhlkH-CVDfFRzcK7y/view
ID=1SxhImd29PLtlvqXAhlkH-CVDfFRzcK7y

echo "--------------------------------------------------------------------------------"
echo "Downloading I-RAVEN dataset into $TARGET_DIR/i-raven"
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

gdown $ID -c -O $TARGET_DIR/temp_i-raven.zip

unzip -q $TARGET_DIR/temp_i-raven.zip -d $TARGET_DIR/i-raven

# if succefully unzipped remove temp file
if [ $? -eq 0 ]; then
    rm $TARGET_DIR/temp_i-raven.zip
fi
