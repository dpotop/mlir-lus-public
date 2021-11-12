#!/bin/bash

./keras-to-pb.py resnet.py
./convert-pb-to-mlir.sh resnet.pb
