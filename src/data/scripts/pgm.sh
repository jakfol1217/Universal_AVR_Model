#!/bin/bash

TARGET_DIR=${1:-"."}  # The directory where the data will be downloaded

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/pgm


echo "--------------------------------------------------------------------------------"
echo "Downloading PGM dataset into $TARGET_DIR/pgm"
echo -e "--------------------------------------------------------------------------------\n"

# URL=https://storage.cloud.google.com/ravens-matrices/attr.rel.pairs.tar.gz
URL=https://storage.googleapis.com/storage/v1/b/ravens-matrices/o/attr.rel.pairs.tar.gz
# https://storage.cloud.google.com/ravens-matrices/attr.rels.tar.gz
# https://storage.cloud.google.com/ravens-matrices/attrs.line.type.tar.gz
# https://storage.cloud.google.com/ravens-matrices/attrs.pairs.tar.gz
# https://storage.cloud.google.com/ravens-matrices/attrs.shape.color.tar.gz
# https://storage.cloud.google.com/ravens-matrices/extrapolation.tar.gz
# https://storage.cloud.google.com/ravens-matrices/interpolation.tar.gz
# https://storage.cloud.google.com/ravens-matrices/neutral.tar.gz

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

wget -c --header="Authorization: Bearer $ACCESS_TOKEN" ${URL}?alt=media -O - | tar -xz -C $TARGET_DIR/pgm
# TODO: rest of urls (too much spaces required)
## Analogies
# https://storage.cloud.google.com/ravens-matrices/analogies/extrapolation.tar.gz
# https://storage.cloud.google.com/ravens-matrices/analogies/interpolation.tar.gz
# https://storage.cloud.google.com/ravens-matrices/analogies/novel.domain.transfer.tar.gz
# https://storage.cloud.google.com/ravens-matrices/analogies/novel.target.domain.line.type.tar.gz
# https://storage.cloud.google.com/ravens-matrices/analogies/novel.target.domain.shape.color.tar.gz
