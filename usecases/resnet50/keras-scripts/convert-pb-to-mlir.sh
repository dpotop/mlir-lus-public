#!/bin/bash

#########################################################
# This script will take a TensorFlow problem file (.pb
# or .pbtxt) and convert it into textual MLIR form.
# Call convention:
#   convert-pb-to-mlir.sh file.pb
#   convert-pb-to-mlir.sh file.pbtxt
# The output is placed into file.mlir
#########################################################
# There is a second way to obtain the same effect, which
# relies on the direct use of TensorFlow commands under
# python. This is in script convert-pb-to-mlir.py.
# The two approaches do not give the same results.
#########################################################


# Turn printing of commands off
set +x

echo "Convert TensorFlow problem file "$1" to MLIR."

# Test if file exists
if [ ! -f $1 ]; then
    echo "   No such file "$1". Exiting..."
    exit
fi

# Extract file name (without path) and extension
filename=$(basename -- "$1")
extension="${filename##*.}"
basename="${filename%.*}"

if [ $extension != "pb" ]; then
    if [ $extension != "pbtxt" ]; then
	echo "   Not a TensorFlow problem file (by extension). Exiting..."
	exit
    fi
fi

newfile=$basename".mlir"

# Actual conversion command
echo "   Start conversion."
tf-mlir-translate --graphdef-to-mlir $1 | tf-opt --tf-executor-island-coarsening --tf-shape-inference > $newfile

echo "   Conversion completed."
