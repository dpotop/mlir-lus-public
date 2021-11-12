#!/usr/bin/env python3

#-----------------------------------------------------------------
# Script used to import the standard ResNet50 network.
# This script can be directly used to produce an MLIR file
# through the command:
#    ../scripts/keras-to-pb.py resnet.py
#    ../scripts/convert-pb-to-mlir.py resnet.pb
# For lisibility, one can also do:
#    ../scripts/strip_mlir.sh resnet.mlir
# The result will be resnet.mlir, resnet_stripped.mlir
#-----------------------------------------------------------------


#-----------------------------------------------------------------
# This code avoids some SSL errors, as as explained in
# https://github.com/tensorflow/tensorflow/issues/33285
#-----------------------------------------------------------------
import requests
requests.packages.urllib3.disable_warnings()
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

#-----------------------------------------------------------------
# Actual importing of the model
#-----------------------------------------------------------------
import tensorflow as tf
keras_model = tf.keras.applications.ResNet50()

