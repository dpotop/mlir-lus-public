## Common configuration

All use cases require the configuration of environment 
variables defining the compilation environment. For the 
**vocoder** use case, this environment includes the C compiler
and linker, and MLIR. For the ML examples, this environment
also includes TensorFlow and IREE.

All these environment variables are defined in 
file ```usecases/makefile.config```. You should open this file
and set the correct values for these variables. 

Assuming that LLVM/MLIR is installed under $(HOME)/llvm, and that you
compile the use cases in the git repository tree, the two variables
that require setting are ```IREE_BIN_DIR``` and ```TF_BIN_DIR```.
These two variables must point in the Bazel compilation cache, where
the binaries are placed. For Linux, this folder is 
under ```$(HOME)/.cache/bazel```. For MacOS, it is 
under ```/private/var/tmp```.

## Coroutine-based vs. sequential implementation

To compile each of the 3 examples, move to the corresponding
folder and execute ```make```. For the execution, each example has 
separate instructions in the README file of its folder.

All three examples can be compiled in two ways:
* Using the coroutine-based method described 
  in [our paper](https://hal.inria.fr/hal-03043623/document). 
  This is the default choice, when ```make``` is invoked.
* By generating a single sequential thread. In this case, instantiated
  nodes are all inlined in the main node. To build this implementation, 
  one must build using the command ```make sequential```.

In the coroutine-based code generation, a run-time is needed.
The application-independent implementation of this run-time is provided 
in the ```usecases/runtime``` folder.

## The use cases
We provide 3 use cases: 
* [A pitch tuning vocoder](vocoder/README.md)
* [A time series CNN based on LSTM](time-series/README.md)
* [An instance of Resnet50](resnet50/README.md)

