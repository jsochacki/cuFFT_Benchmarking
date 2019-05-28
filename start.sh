#!/bin/bash
set -x
if [ "$#" -ne 1 ]; then
   echo "Usage: ./start.sh <EXECUTIBLE_NAME>"
   exit
fi

EXECUTIBLE_NAME=$1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
./$EXECUTIBLE_NAME