#!/bin/bash

echo ">> Making packages: $pkg_names"
if [ -z "$pkg_names" ]
then 
	echo !! Package names not specified.
	exit 99
fi

for pkg_name in $pkg_names
do
	path=${pkg_name}.tar.gz
	if [ -e $path ]
	then
		rm $path
	fi
	tar -czf $path ../${pkg_name}
done
