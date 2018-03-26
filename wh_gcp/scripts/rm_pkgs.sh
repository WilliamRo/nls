#!/bin/bash
# SYNTAX:
#   rm_pkgs pkg_dir pkg_1 pkg_2 ...

# Check inputs	
if [[ $# -lt 2 ]]; then 
	echo '!! Package directory and package' \
	     'names must be specified'
	exit 9
fi
pkg_dir=$1
shift

# Remove directories
while [[ $# -gt 0 ]]; do 
	tgt=${pkg_dir}/$1
	rm -rf $tgt
	echo ">> Package $tgt removed"
	shift
done


