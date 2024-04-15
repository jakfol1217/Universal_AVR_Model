#!/bin/bash

TARGET_DIR=${1:-"."}  # The directory where the data will be downloaded

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/raw
mkdir -p $TARGET_DIR/vasr

URL=https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip
URL_TRAIN_LABELS=https://drive.google.com/file/d/1bTbRfF8UxD8gwu41K3UGIro4iTEz12Zf/view
ID_TRAIN=1bTbRfF8UxD8gwu41K3UGIro4iTEz12Zf
URL_DEV_LABELS=https://drive.google.com/file/d/1q6KBXXxWcr3d8HetuJtqSYZ6FBq2Ofks/view
ID_DEV=1q6KBXXxWcr3d8HetuJtqSYZ6FBq2Ofks
# Check if unzip and gdown is installed if not install it

if ! [ -x "$(command -v unzip)" ]; then
    echo 'Error: unzip is not installed.' >&2
    exit 1
fi

if ! [ -x "$(command -v gdown)" ]; then
    echo 'Error: gdown is not installed.' >&2
    exit 1
fi

echo "--------------------------------------------------------------------------------"
echo "Downloading VASR dataset into $TARGET_DIR/vasr"
echo -e "--------------------------------------------------------------------------------\n"

wget -c $URL -O $TARGET_DIR/raw/vasr_images.zip

unzip -q $TARGET_DIR/raw/vasr_images.zip -d $TARGET_DIR/vasr

echo "--------------------------------------------------------------------------------"
echo "Downloading VASR dataset annotations into $TARGET_DIR/vasr"
echo -e "--------------------------------------------------------------------------------\n"

gdown $ID_TRAIN -c -O $TARGET_DIR/vasr/train.csv

gdown $ID_DEV -c -O $TARGET_DIR/vasr/dev.csv
