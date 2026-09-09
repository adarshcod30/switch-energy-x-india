#!/bin/bash
# usage: submit.sh <file> "<message>"
python3 -m kaggle competitions submit -c switch-energy-x-india -f "$1" -m "$2" 2>&1 | tail -3
