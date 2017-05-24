#!/bin/bash

curdir=`pwd`
dir='.'

while getopts "l:" arg
do
    case $arg in
		l)
			if [[ ! -d $OPTARG ]]; then
				echo "Please input a folder location -l"
				exit 0
			fi
			dir=$OPTARG
            ;;
        *)
            echo "Usage $0 [-l folder]"
            exit 0
    esac
done

if [ -n "${DNORM_HOME##+([[:space:]])}" ]; then
	cd $DNORM_HOME
fi

find $dir -type f | while read file
do
	RunDNorm.sh config/banner_NCBIDisease_TEST.xml data/CTD_diseases.tsv output/simmatrix_NCBIDisease_e4.bin $file "$dir/`basename "$file" .txt`.dnorm.txt"
done

cd $curdir