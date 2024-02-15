#!/bin/bash

TARGET_DIR=${1:-"."}  # The directory where the data will be downloaded

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/deepiq

URL=https://github.com/deepiq/deepiq/raw/master/odd-one-out%20test%20examples.zip

# Check if unzip is installed

if ! [ -x "$(command -v unzip)" ]; then
    echo 'Error: unzip is not installed.' >&2
    exit 1
fi

echo "--------------------------------------------------------------------------------"
echo "Downloading Deepiq OOO dataset into $TARGET_DIR/deepiq"
echo -e "--------------------------------------------------------------------------------\n"


wget -c $URL -O $TARGET_DIR/deepiq.zip

unzip -j -q $TARGET_DIR/deepiq.zip -d $TARGET_DIR/deepiq

# if succefully unzipped remove temp file

if [ $? -eq 0 ]; then
    rm $TARGET_DIR/deepiq.zip
fi


