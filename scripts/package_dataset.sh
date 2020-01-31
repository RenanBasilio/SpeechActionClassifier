#!/bin/bash

if [ -n "$1" ]; then
    echo -e "Packaging dataset..."
    cd "$1"
    SIZE=`du -skc unsorted classes.txt splits.txt classes.info | sed '$!d' | cut -f 1`
    tar cf - unsorted classes.txt splits.txt classes.info -P | pv -s ${SIZE}k | gzip > D3PERJ-COPPETEC-$(date +'%Y-%m-%d').tar.gz
    echo -e "Done."
else
    echo -e "Usage: package_dataset.sh <dir>"
fi