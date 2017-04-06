#!/bin/bash

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


skrmedpostctl start && wsdserverctl start

find $dir -type f | while read file
do	
	metamap $file "$dir/`basename "$file" .txt`.mesh.txt"
done
