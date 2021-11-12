#!/usr/bin/env python3

#---------------------------------------------------------------
# Transform a keras trained model into a frozen graph allowing
# embedded implementation. Then, save the problem file and
# also convert it to MLIR and save the MLIR file.
#
# The definition (and the training) of the model must be done
# in a separate script that must be imported here. Otherwise,
# the conversion of variables into constants complains that it
# cannot find the source code of the model.
#
# The script takes as argument the name of the Python script
# that builds the model. This script must define variable
# keras_model, which contains the model that is converted.
#
# NOTE: For MLIR code generation, it is assumed that the Keras
# model has only one input and only one output. All models
# produced using the sequential approach are of this type.
#---------------------------------------------------------------

print('keras-to-pb.py execution started.')

import sys
if len(sys.argv)<=1 :
    print('  Missing keras model name. Exiting...')
    sys.exit(0)
fullname = sys.argv[1]
print('  Model input file: ',fullname)

import os
fullname = os.path.abspath(fullname)
dirname = os.path.dirname(fullname)
# Allow python to find the file as a module by
# inserting itto sys.path in the second position
# (the first one is taken).
sys.path.insert(1,dirname)    

filename = os.path.basename(fullname)
modelname = os.path.splitext(filename)[0]
if filename != (modelname+".py") :
    print('  Input file does not have .py extension. Exiting...')
    sys.exit(0)
print('  Model name (extracted from file name):',modelname)

print('  Importing the model ',modelname,' of file ',filename)
import importlib
mod = importlib.import_module(modelname)
print('  Model import complete.')
try:
    keras_model = mod.keras_model
except AttributeError:
    print('  Imported model does not define the keras_model variable/attribute. Exiting...')
    sys.exit(0)

print('  Freeze the model.')
import tensorflow as tf
full_model = tf.function(lambda x: keras_model(x))
fm = full_model.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
frozen_func = convert_variables_to_constants_v2(fm)

# To create a .pbtxt file, change the file extension string and
# change "as_text=False" into "as_text=True"
print('  Save the frozen model as ',modelname+'.pb')
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir='', # path
                  name=(modelname+'.pb'), #filename
                  as_text=False)

# 
# print('  Convert the frozen model to MLIR')
# mlir_code = tf.mlir.experimental.convert_function(fm)
# print('  Save the MLIR code to file ',modelname+'.mlir')
# mlir_file = open(modelname+".mlir","w")
# mlir_file.write(mlir_code)
# mlir_file.close()

print('  Done.')
# print(x)


# Can also print the graph this way
# frozen_func.graph.as_graph_def()
# Can summarize the graph this way
# layers = [op.name for op in frozen_func.graph.get_operations()]
# print("-" * 60)
# print("Frozen model layers: ")
# for layer in layers:
#     print(layer)print("-" * 60)
# print("Frozen model inputs: ")
# print(frozen_func.inputs)
# print("Frozen model outputs: ")
# print(frozen_func.outputs)


