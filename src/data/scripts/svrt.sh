#!/bin/bash

TARGET_DIR=${1:-"."}  # The directory where the data will be downloaded

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/svrt

echo "--------------------------------------------------------------------------------"
echo "Generating SVRT dataset into $TARGET_DIR/svrt"
echo -e "--------------------------------------------------------------------------------\n"

# Check generation files exists
SCRIPT_PATH=$(realpath $0)
DATA_ROOT=$(dirname $(dirname $SCRIPT_PATH))

if [ ! -f $DATA_ROOT/svrt-HEAD-78330d7.tar.gz ]; then
    echo "Download svrt repo snapshot first (https://fleuret.org/cgi-bin/gitweb/gitweb.cgi?p=svrt.git;a=snapshot;h=HEAD;sf=tgz)"
    echo "and place it in $DATA_ROOT directory."
    exit 1
fi

tar -xzf $DATA_ROOT/svrt-HEAD-78330d7.tar.gz

OLD_PWD=$(pwd)

cd $DATA_ROOT/svrt-HEAD-78330d7

# apt-get install libjpeg-dev
# apt-get install libpng-dev
$DATA_ROOT/svrt-HEAD-78330d7/doit.sh

if [ $? -ne 0 ]; then
    echo "Error while generating SVRT dataset" >&2
    echo "One possible problem is that the required libraries are not installed (apt-get install libjpeg-dev | libpng-dev)" >&2
    exit 1
fi

mv $DATA_ROOT/svrt-HEAD-78330d7/results_problem_{1..23} $TARGET_DIR/svrt

# tidying up
cd $OLD_PWD
rm -rf $DATA_ROOT/svrt-HEAD-78330d7
