#!/bin/bash

echo "[Local] Configuring wh_gcp demo ..."

DATA_PATH=../data/wiener_hammerstein/whb.tfd
LOCAL_PYTHON='/home/walienluo/.interpreter/python3/bin/python3.6'
gcloud config set ml_engine/local_python $LOCAL_PYTHON

echo "[Local] wh_gcp configuration has been Done!"
