#!/bin/bash


usage() {
  echo "Usage: $0 [-r REGIME] [-h] [TARGET_DIR]"
  echo ""
  echo "Download VAP dataset"
  echo "  -r REGIME  The regime of the VAP dataset. Options: extrapolation, interpolation, novel.domain.transfer, novel.target.domain.line.type, novel.target.domain.shape.color"
  echo "  -h         Display this help and exit"
}

# read flags from the command line
while getopts 'h:r:' opt; do
case "${opt}" in
    r) REGIME="${OPTARG}" ;;
    h) usage ; exit 0 ;;
    *) usage ; exit 1 ;;
esac
done

if [[ ! $REGIME = @(extrapolation|interpolation|novel.domain.transfer|novel.target.domain.line.type|novel.target.domain.shape.color) ]]; then
    echo "Invalid regime. Options: extrapolation, interpolation, novel.domain.transfer, novel.target.domain.line.type, novel.target.domain.shape.color"
    exit 1
fi


TARGET_DIR=${@:$OPTIND:1:} # The directory where the data will be downloaded
# Add default value to TARGET_DIR
if [ -z "$TARGET_DIR" ]; then
    TARGET_DIR="."
fi

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/raw
mkdir -p $TARGET_DIR/vap


echo "--------------------------------------------------------------------------------"
echo "Downloading VAP dataset into $TARGET_DIR/vap"
echo -e "--------------------------------------------------------------------------------\n"

# https://console.cloud.google.com/storage/browser/ravens-matrices/analogies
URL=https://storage.googleapis.com/ravens-matrices/analogies/${REGIME}.tar.gz

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

wget -c --header="Authorization: Bearer $ACCESS_TOKEN" ${URL}?alt=media -O ${TARGET_DIR}/raw/vap_${REGIME}.tar.gz
tar -xzf ${TARGET_DIR}/raw/vap_${REGIME}.tar.gz ${TARGET_DIR}/vap
