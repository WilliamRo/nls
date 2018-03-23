#!/bin/bash
# SYNTAX:
#   bash cp_pkgs.sh src_dir dest_dir pkg1 pkg2 ...

# Check inputs
if [[ $# -lt 3 ]]; then
	echo '!! Not enough intput arguments'
	exit 9
fi
src_dir=$1
dest_dir=$2
shift 2

# Copy packages
while [[ $# -gt 0 ]]; do
	src=${src_dir}/$1
	dest=${dest_dir}/$1
	# Check source
	if [[ ! -d $src ]]; then
		echo "!! Can not find package $src"
		exit 9
	fi
	# Check destination
	if [[ -d $dest_dir ]]; then 
		bash scripts/rm_pkgs.sh $dest_dir $1
	fi
	# Copy package
	mkdir $dest
	cp -r ${src}/* $dest
	if [[ ! $? ]]; then
		echo ">> Failed to copy package"
		exit 9
	fi
	echo ">> Package $src copied to $dest"

	shift
done 


