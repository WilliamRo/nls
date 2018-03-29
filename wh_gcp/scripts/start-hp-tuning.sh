#!/bin/bash

# Default configuration
MODEL_MARK=mlp
JOB_NAME=$MODEL_MARK
BUCKET_NAME=nls-nn-bucket
GS_ROOT=gs://$BUCKET_NAME
REGION=us-central1
PACKAGE_NAME=trainer
POSTFIX=$(date -u +%d%H%M)

epoch=3
batch_size=64

# Prepare packages
pkg_names='tframe models signals'
bash scripts/cp_pkgs.sh .. . $pkg_names 

# Parse arguments manually
while [ $# -gt 0 ]
do
	case $1 in 
		--epoch)
			epoch=$2
			shift 2
			;;
		--job-name)
			JOB_NAME=$2
			shift 2
			;;
		*)
			echo !! Can\'t resolve flag $1, process aborted.
			exit 9
			;;
	esac
done

# Update job name
JOB_NAME=${JOB_NAME}_$POSTFIX
OUTPUT_PATH=${GS_ROOT}/hpt/$JOB_NAME

# Show status
echo '>> Start to run a single-instance trainer in the cloud ...'
echo :: Configurations:
echo ... job name:   $JOB_NAME
echo ... epoch:      $epoch
echo ... batch size: $batch_size

#exit 0

# Clear gs://.../packages

#--packages ../tframe/,../models/,../signals/ \
gcloud ml-engine jobs submit training $JOB_NAME \
	--job-dir $OUTPUT_PATH \
	--runtime-version 1.6 \
	--config hptuning_config.yaml \
	--package-path ${PACKAGE_NAME}/ \
	--module-name ${PACKAGE_NAME}.hpt_task \
	--region $REGION \
	-- \
	--mark $MODEL_MARK \
	--epoch $epoch \
	--data_dir ${GS_ROOT}/data/whb/whb.tfd \
  --hpt True

# Clear path
bash scripts/rm_pkgs.sh . $pkg_names

