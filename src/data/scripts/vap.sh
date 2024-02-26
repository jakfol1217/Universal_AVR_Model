#!/bin/bash

TARGET_DIR=${1:-"."}  # The directory where the data will be downloaded

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/vap


echo "--------------------------------------------------------------------------------"
echo "Downloading VAP dataset into $TARGET_DIR/vap"
echo -e "--------------------------------------------------------------------------------\n"

# https://console.cloud.google.com/storage/browser/ravens-matrices/analogies
URL=https://storage.googleapis.com/storage/v1/b/ravens-matrices/o/extrapolation.tar.gz
# URL=https://storage.googleapis.com/storage/v1/b/ravens-matrices/o/interpolation.tar.gz
# URL=https://storage.googleapis.com/storage/v1/b/ravens-matrices/o/novel.domain.transfer.tar.gz
# URL=https://storage.googleapis.com/storage/v1/b/ravens-matrices/o/novel.target.domain.line.type.tar.gz
# URL=https://storage.googleapis.com/storage/v1/b/ravens-matrices/o/novel.target.domain.shape.color.tar.gz

# check if environment variables are set

if [ -z "$GOOGLE_CLIENT_ID" ]; then
    echo "Please set the GOOGLE_CLIENT_ID environment variable"
    exit 1
fi

if [ -z "$GOOGLE_CLIENT_SECRET" ]; then
    echo "Please set the GOOGLE_CLIENT_SECRET environment variable"
    exit 1
fi

# check if jq command exists
if ! [ -x "$(command -v jq)" ]; then
    echo "Please install jq"
    exit 1
fi

CLIENT_ID=${GOOGLE_CLIENT_ID}
CLIENT_SECRET=${GOOGLE_CLIENT_SECRET}
REDIRECT_URI=${GOOGLE_REDIRECT_URI:-"http://127.0.0.1"}

echo -e "Please enter the code.\nYou can obtain it from https://accounts.google.com/o/oauth2/auth?client_id=${CLIENT_ID}&redirect_uri=${REDIRECT_URI}&scope=https://www.googleapis.com/auth/devstorage.read_only&email&response_type=code&include_granted_scopes=true&access_type=offline&state=state_parameter_passthrough_value)\n"
read CODE

ACCESS_TOKEN=$(curl -X POST https://oauth2.googleapis.com/token \
    -d "code=${CODE}&client_id=${CLIENT_ID}&client_secret=${CLIENT_SECRET}&redirect_uri=${REDIRECT_URI}&access_type=offline&grant_type=authorization_code" \
    | jq -r '.access_token')
echo $ACCESS_TOKEN

wget -c --header="Authorization: Bearer $ACCESS_TOKEN" ${URL}?alt=media -O - | tar -xz -C $TARGET_DIR/vap
# TODO: rest of urls (too much spaces required)
