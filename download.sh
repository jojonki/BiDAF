#!/bin/sh

DATA_DIR="./dataset/"
mkdir -p $DATA_DIR

download () {
  URL=$1
  FILE_NAME=$2

  if [ ! -f "$DATA_DIR$FILE_NAME" ]; then
    wget $URL$FILE_NAME -O $DATA_DIR/$FILE_NAME
  else
    echo "You've already downloaded $FILE_NAME dataset"
  fi
}


download "https://rajpurkar.github.io/SQuAD-explorer/dataset/" "train-v1.1.json"
download "https://rajpurkar.github.io/SQuAD-explorer/dataset/" "dev-v1.1.json"