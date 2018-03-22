#!/bin/bash

# Default configuration
JOB_NAME=nls_nn
BUCKET_NAME=nls-nn-bucket
STAGING_BUCKET=gs://nls-nn-storage
OUTPUT_PATH=gs://$BUCKET_NAME/
REGION=us-central1
PACKAGE_NAME=trainer

epoch=2
batch_size=64

# Prepare packages
pkg_names='tframe models signals'
export pkg_names
export PACKAGE_NAME
bash scripts/make_packages.sh
if [ $? -ne 0 ]
then 
	exit 1
fi
PACKAGES=""
counter=1
for pkg in $pkg_names
do
	if [[ $counter -gt 1 ]]; then
		PACKAGES=${PACKAGES},
	fi
	PACKAGES=${PACKAGES}${pkg}.tar.gz
	((counter++))
done

# Parse arguments manually
while [ $# -gt 0 ]
do
	case $1 in 
		--epoch)
			epoch=$2
			shift 2
			;;
		--batch_size)
			batch_size=$2
			shift 2
			;;
		--job-name)
			JOB_NAME=$2
			shift 2
			;;
		*)
			echo !! Can\'t resolve flag $1, process aborted.
			exit 99
			;;
	esac
done

JOB_NAME=${JOB_NAME}_$(date -u +%y%m%d%H%M%S)

# Show status
echo '>> Start to run a single-instance trainer in the cloud ...'
echo :: Configurations:
echo ... job name:   $JOB_NAME
echo ... epoch:      $epoch
echo ... batch size: $batch_size
#echo ... $PACKAGES

#exit 0

#--packages ../tframe/,../models/,../signals/ \
gcloud ml-engine jobs submit training $JOB_NAME \
	--job-dir $OUTPUT_PATH \
	--runtime-version 1.5 \
	--package-path ${PACKAGE_NAME}/ \
	--packages $PACKAGES \
	--module-name ${PACKAGE_NAME}.task \
	--region $REGION \
	-- \
	--mark $JOB_NAME \
	--epoch $epoch \
	--batch_size $batch_size



