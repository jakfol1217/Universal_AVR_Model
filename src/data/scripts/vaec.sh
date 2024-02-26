#!/bin/bash

TARGET_DIR=${1:-"."}  # The directory where the data will be downloaded

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/vaec

echo "--------------------------------------------------------------------------------"
echo "Generating VAEC dataset into $TARGET_DIR/vaec"
echo -e "--------------------------------------------------------------------------------\n"

# Check generation files exists
SCRIPT_PATH=$(realpath $0)
DATA_ROOT=$(dirname $(dirname $SCRIPT_PATH))

if [ ! -f $DATA_ROOT/Extrapolation/VAEC_dataset_and_models/dset_gen/VAEC_scale_extrap.py ] || [ ! -f $DATA_ROOT/Extrapolation/VAEC_dataset_and_models/dset_gen/VAEC_trans_extrap.py ]; then
    echo "Download Extrapolation submodules first"
    exit 1
fi

# Check if python venv exists if not create it
if [ ! -d $DATA_ROOT/Extrapolation/VAEC_dataset_and_models/dset_gen/.venv ]; then
    echo "Creating python venv"
    python3 -m venv $DATA_ROOT/Extrapolation/VAEC_dataset_and_models/dset_gen/.venv
    source $DATA_ROOT/Extrapolation/VAEC_dataset_and_models/dset_gen/.venv/bin/activate
    pip install -r $DATA_ROOT/Extrapolation/VAEC_dataset_and_models/dset_gen/requirements.txt
fi

OLD_PWD=$(pwd)
mkdir -p $TARGET_DIR/vaec/temp
cd $TARGET_DIR/vaec/temp
cp $DATA_ROOT/Extrapolation/VAEC_dataset_and_models/util.py ../util.py

python3 $DATA_ROOT/Extrapolation/VAEC_dataset_and_models/dset_gen/VAEC_scale_extrap.py
python3 $DATA_ROOT/Extrapolation/VAEC_dataset_and_models/dset_gen/VAEC_trans_extrap.py

# tidying up
cd $OLD_PWD
rm -rf $TARGET_DIR/vaec/temp
rm $TARGET_DIR/vaec/util.py

# Deactivate python venv
source $DATA_ROOT/Extrapolation/VAEC_dataset_and_models/dset_gen/.venv/bin/activate

# Remove created virtual environment
rm -rf $DATA_ROOT/Extrapolation/VAEC_dataset_and_models/dset_gen/.venv
