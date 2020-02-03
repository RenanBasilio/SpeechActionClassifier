#!/bin/bash

if [ -n "$1" ]; then
    echo -e "Packaging dataset..."
    cwd=$PWD
    cd "$1"
    SIZE=`du -skc unsorted classes.txt splits.txt dataset.info | sed '$!d' | cut -f 1`
    tar cf - unsorted classes.txt splits.txt dataset.info -P | pv -s ${SIZE}k | gzip > $cwd/D3PERJ-COPPETEC-$(date +'%Y-%m-%d').tar.gz
    echo -e "Done."
else
    echo -e "Usage: package_dataset.sh <dir>"
fi