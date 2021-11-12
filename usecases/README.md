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

## The use cases
We provide 3 use cases: 
* [A pitch tuning vocoder](usecases/vocoder/README.md)
* [A time series CNN based on LSTM](usecases/time-series/README.md)
* [An instance of Resnet50](usecases/resnet50/README.md)

