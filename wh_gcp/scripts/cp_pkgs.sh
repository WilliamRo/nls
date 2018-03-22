#!/bin/bash

# Check variables
if [[ -n "$pkg_names" ]]; then
	echo !! Package names not specified.
	exit 99
fi
if [[ -n "$PACKAGE_NAME" ]]; then
	echo !! Package name note specified.
	exit 98
fi

# Copy packages
for pkg in $pkg_names
do
	src_dir=../${pkg}/*
	dest_dir=${PACKAGE_NAME}/$pkg
	# Check the existence of pkg
done


