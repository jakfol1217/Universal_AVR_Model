#!/bin/bash


usage() {
  echo "Usage: $0 [-r REGIME] [-h] [TARGET_DIR]"
  echo ""
  echo "Download PGM dataset"
  echo "  -r REGIME  The regime of the PGM dataset. Options: attr.rel.pairs, attr.rels, attrs.line.type, attrs.pairs, attrs.shape.color, extrapolation, interpolation, neutral"
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

if [[ ! $REGIME = @(attr.rel.pairs|attr.rels|attrs.line.type|attrs.pairs|attrs.shape.color|extrapolation|interpolation|neutral) ]]; then
    echo "Invalid regime. Options: attr.rel.pairs, attr.rels, attrs.line.type, attrs.pairs, attrs.shape.color, extrapolation, interpolation, neutral"
    exit 1
fi

TARGET_DIR=${@:$OPTIND:1:}  # The directory where the data will be downloaded
# Add default value to TARGET_DIR
if [ -z "$TARGET_DIR" ]; then
    TARGET_DIR="."
fi

# Create the target directory if it doesn't exist
mkdir -p $TARGET_DIR/raw
mkdir -p $TARGET_DIR/pgm


echo "--------------------------------------------------------------------------------"
echo "Downloading PGM dataset into $TARGET_DIR/pgm"
echo -e "--------------------------------------------------------------------------------\n"

URL=https://storage.googleapis.com/storage/v1/b/ravens-matrices/o/${REGIME}.tar.gz

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

wget -c --header="Authorization: Bearer $ACCESS_TOKEN" ${URL}?alt=media -O ${TARGET_DIR}/raw/pgm_${REGIME}.tar.gz
tar -xzf ${TARGET_DIR}/raw/pgm_${REGIME}.tar.gz ${TARGET_DIR}/pgm
