#!/bin/bash

if [ "$#" -eq 0 ]; then
   echo "Usage: ./start.sh <EXECUTIBLE_NAME> <ARGUEMENT_1> <ARGUEMENT_2> etc..."
   exit
fi

ARGUEMENTS=
EXECUTIBLE_NAME=

for var in "$@"
do
   if [ -z "$EXECUTIBLE_NAME" ]; then
      EXECUTIBLE_NAME=$var
   else
      ARGUEMENTS+=" ${var}"
   fi
done

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
./$EXECUTIBLE_NAME$ARGUEMENTS