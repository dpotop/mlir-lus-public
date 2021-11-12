Resnet50 is a convolutional neural network which is 50 layers deep which
performs computationnally-intensive operations.

### Resnet50 as our usecase

The directories usecases/resnet50 (on which we performed our benchmarks and 
which splits lus and tensorflow code in two files) and
usecases/resnet50-compact (which provide an implementation combining lus and
tensorflow code in the same file) contain Resnet50 as our usecases.

### Build the use case

Move to the ```resnet50``` directory and build the use case ```make```
The result of compilation is file ```resnet```. It prints the timestamp at
each cycle.

### Production of Resnet50

We provide a full (trained) Resnet50 implementation, but you can produce it 
yourself, from the Python Tensorflow library. You have to launch the 
usecases/produce-resnet50/produce-resnet50.sh script. In the produced 
resnet.mlir file, you just have to :
- Manually remove the wrapping operations tf\_executor.graph and 
  tf\_executor.island
- Manually replace the value produced by the "tf.Placeholder" op by a function
  parameter.
- Manually return the last value produced.
- Update the function type.

Then you can use this file instead of usecases/resnet50/resnet.tf.mlir.
If you (manually) change the function into a lus node, you can use it instead 
of usecases/resnet50-compact/resnet.lus.mlir.
