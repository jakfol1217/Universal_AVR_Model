#!/bin/bash

TARGET_DIR=${1:-"."}  # The directory where the data will be downloaded

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/dopt

echo "--------------------------------------------------------------------------------"
echo "Generating DOPT dataset into $TARGET_DIR/dopt"
echo -e "--------------------------------------------------------------------------------\n"

# Check generation files exists
SCRIPT_PATH=$(realpath $0)
DATA_ROOT=$(dirname $(dirname $SCRIPT_PATH))

if [ ! -f $DATA_ROOT/Extrapolation/dynamic_object_prediction/generate_data.py ]; then
    echo "Download Extrapolation submodules first"
    exit 1
fi

# Check if python venv exists if not create it
if [ ! -d $DATA_ROOT/Extrapolation/dynamic_object_prediction/.venv ]; then
    echo "Creating python venv"
    python3 -m venv $DATA_ROOT/Extrapolation/dynamic_object_prediction/.venv
    source $DATA_ROOT/Extrapolation/dynamic_object_prediction/.venv/bin/activate
    pip install -r $DATA_ROOT/Extrapolation/dynamic_object_prediction/requirements-generator.txt
fi

OLD_PWD=$(pwd)
cd $TARGET_DIR/dopt
python $DATA_ROOT/Extrapolation/dynamic_object_prediction/generate_data.py
cd $OLD_PWD

# Deactivate python venv
source $DATA_ROOT/Extrapolation/dynamic_object_prediction/.venv/bin/activate

# Remove created virtual environment
rm -rf $DATA_ROOT/Extrapolation/dynamic_object_prediction/.venv
