# Installation instructions

## External packages
Our tool is built on top of LLVM/MLIR, which itself requires:
* A recent version of clang (we recommend clang-9). For Ubuntu Linux, it can 
  be installed with:
  ```
  sudo apt-get install clang-9 llvm-9.0 llvm-devel clang-devel
  ```
* A recent version of cmake (>= cmake 13). For Ubuntu Linux, it must be 
  installed from sources to be downloaded at ```https://cmake.org/download```
* Python 3 and pip :
  ```
   sudo apt-get install python3 python3-dev python3-pip
   ```
* The python TensorFlow library :
   ```
   pip install tensorflow
   ```
* A recent version of Bazel, which installation on Ubuntu Linux can be 
  performed as described 
  [here](https://docs.bazel.build/versions/main/install-ubuntu.html).
* Recent versions of libgmp and z3 (can be found on opam)

On MacOSX using macports, in some cases, full macports removal and 
reinstallation may be needed. This can be done following the instructions
at: https://trac.macports.org/wiki/Migration

## LLVM/MLIR
### Clone The LLVM Project repo and select the good version
```
git clone https://github.com/llvm/llvm-project
git checkout bcd6424f9b693af57b29a0f03c52d6991be35d41
```
### Compilation and installation
We make the assumption that installation prefix is $(HOME)/llvm. 
After compilation, ```$HOME/llvm/bin``` must be added to $PATH.
- On Linux 64 bits :
```
cd llvm-project
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD=AArch64 -DCMAKE_INSTALL_PREFIX=$HOME/llvm \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_ASM_COMPILER=clang \
  -DCMAKE_SHARED_LINKER_FLAGS="-fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fno-omit-frame-pointer" ../llvm
make -j4
make install
```
- On MacOs M1 :
```
cd llvm-project
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="ARM;X86;AArch64" \
  -DLLVM_DEFAULT_TARGET_TRIPLE="arm64-apple-darwin20.6.0" \
  -DCMAKE_OSX_ARCHITECTURES='arm64' -DCMAKE_INSTALL_PREFIX=$HOME/llvm \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_ASM_COMPILER=clang \
  -DCMAKE_SHARED_LINKER_FLAGS="-fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fno-omit-frame-pointer" ../llvm
make -j8
make install
```
	
## IREE/Tensorflow
### Clone the IREE repo
```
git clone https://github.com/google/iree.git
git checkout 4f218a8be5ba3e840ebfc8300c9124d01ab0ecc1
git submodule update --init
```
### Compilation of Tensorflow
From iree directory :
```
cd third_party/tensorflow
git submodule update --init
bazel build //tensorflow/compiler/mlir:tf-mlir-translate
bazel build //tensorflow/compiler/mlir:tf-opt
```
After compilation, 
```iree/third_party/tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/mlir```
must be added to $PATH.
### Compilation of IREE
From iree directory :
```
python3 configure_bazel.py
bazel build //tensorflow/compiler/mlir:tf-mlir-translate
bazel build //iree/tools:iree-opt
```
   
After compilation, ```iree/bazel-out/k8-opt/bin/iree/tools``` must be added 
to $PATH.

## Sound eXchange (SoX)
Our use case requires the use of the sox sound processing toolbox. 
* On Ubuntu Linux, it can be installed with ```sudo apt-get install sox```
* Under MacOSX, using ```macports```, the install command 
  is ```sudo port install sox```

## MLIRLus

MLIRLus is the implementation of our embedding of Lustre in MLIR.

### Compilation

In the mlir-lus folder :

1- Create the file Makefile.config and set within it the variables LLVM, CPP, 
   LDFLAGS, CC. A standard one for Ubuntu can be this one :
```
LLVM=$(HOME)/llvm
CPP=clang++-9
LDFLAGS= -L $(LLVM)/lib -lpthread -g
CC=clang-9
LD=clang++-9
```

2- Build dependencies with ```touch makefile.depend```, 
   followed by ```make depend```

3- Build mlirlus with ```make```

### Options

The produced binary (named mlirlus) takes an input file and prints its output
on the standard output. The command : ```./mlirlus file.lus``` just parses
file.lus, verify its correction and prints it.
Several options can be composed :
* ```--normalize``` replaces fby and pre operations by representing state
  in the node signature, and ensures that dominance rules are now respected
  in lus code.
* ```--inline-nodes``` inline nodes instead of truly instantiating them when
  lus.instance is specified.
* ```--convert-lus-to-sync``` convert lus operations to sync operations and
  scf operations (for the main loop and for the clock-based predicates).
* ```--convert-sync-to-scf``` convert sync operations to std operations

## MLIR prime

MLIR prime is an auxiliary tool which implements a bunch of optimizations
using MLIR libraries (and some bug corrections).

### Compilation

In the mlir-prime folder :

1- Create the file Makefile.config and set within it the variables LLVM, CPP, 
   LDFLAGS, CC. A standard one for Ubuntu can be this one :
```
LLVM=$(HOME)/llvm
CPP=clang++-9
LDFLAGS= -L $(LLVM)/lib -lpthread -g
CC=clang-9
LD=clang++-9
```

2- Build dependencies with ```touch makefile.depend```, 
   followed by ```make depend```

3- Build mlir-prime with ```make```

### Options

The produced binary (named mlir-prime) takes an input file and prints its output
on the standard output. The command : ```./mlir-prime file.prime``` just parses
file.prime, verify its correction and prints it.
Several options can be composed with the builtin options of MLIR :
* ```--bufferize-linalg-reshape``` corrects a limitation of the bufferization
  passes on the version of MLIR we used for the use cases (this limitation
  is corrected in the current MLIR version).
* ```--remove-copy-prime``` removes linalg.copy operations on which we had
  problems for lowering.
* ```--prime-linalg-to-affine``` bypasses the builtin lowering of the dialect 
  linalg to the dialect affine using packing, tiling... These optimisations
  are machine-dependent and correspond to our benchmark architecture.
* ```--loop-permutation-prime``` extends the built-in ```loop-permutation```
  option which just operates on the first loop of the program. We are
  currently submitting this extension to the MLIR project.

## Usecases

For each usecase, if needed, change variables on top of its Makefile. Example :
variable ```MLIRLUS ``` must be set to the  path of your build of 
```mlir-lus``` (normally ```../../mlir-lus```).

### Pitch tuning vocoder

### Build the use case

Move to the ```pitch-sched``` directory and build the use case ```make```
The result of compilation is file ```pitch```.

### Running the use case

To allow execution, the computer must have a sound input and 
a sound output device. On a common notebook, these are either 
the built-in microphone and speakers, or a headset. The second
variant is recommended, as it avoids sound feed-back (Larsen
effect).

It is assumed that SoX has been configured to use these as
default input and output devices. This can be tested by running 
the following command, which should result in an echo:
```
   rec -q -t raw -r 44100 -e signed -b 16 -c 2 - | \
                play -q -t raw -r 44100 -e signed -b 16 -c 2 -
```

To launch execution, execute ```make run```.  The application will
continuously capture the input from the standard sound device and send
the processed output to the standard sound device. When you talk into
the microphone, the output will be your voice with a changed pitch (by
default, higher by 3 semitones). If the result is an increasing noise,
it is probable that you do not have a headset, and that a feedback
(Larsen effect) takes place. The solution is to reduce the system
volume (or use a headset).

To stop execution, press ```<CTRL>-C```, followed by ```make kill```.

During execution, one can change:
* The volume, by 10% increments, using keys ```+``` and ```-```
      (one positive or negative increment by key).
* The pitch shift, by half-semitone increments, using
      keys ```n``` (positive increment) or ```m``` (negative increment)
* Mute the sound using key ```s```
* Unmute the sound using key ```a```

To provide a command key, write it, and then press ```<ENTER>```.
    Multiple command keys can be placed on a single line
    before pressing ```<ENTER>```. For instance, writing:

	mmmmmmm<ENTER>
		
will reduce the pitch by 3.5 semitones.
   
We use this (primitive) interface in order to avoid further
    reliance on external libraries such as ncurses.

### Description of the use case

The source code of the use case consists of files of 3 types:
* ```lus``` dialect code is grouped in file ```pitch.lus.mlir```, 
  where it is mixed with regular MLIR code of the ```Standard```
  dialect. 

* Regular MLIR code is grouped in the other ```*.mlir``` files.
  This comprises the majority of the use case. It can be 
  integrated with the reactive code, but we preferred keeping
  the reactive code apart for lisibility.
  
* C code:
  * The implementation of the longjmp-based scheduler.
  * ```sndio.c``` provides the soundcard read and write routines,
    which cannot be written in MLIR in the standard dialect (only
	in the lower-level LLVM dialect).
  * ```bitrev.c``` is the only only function written in C
    that does not interface with the OS. It can be
    converted to MLIR, but we preferred showcasing the ability 
	to interface.

## Resnet50

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

## Timeseries

This Recurrent Neural Network with one single LSTM layer is the one described
in the paper. We reimplemented it on the basis of the method used for Resnet50,
but it needed more significant modifications (about the steate representation).
We also reduce the learnt-weigths to zero constant tensors.

### Timeseries as our usecase

The directory usecases/time-series provides our implementation interfaced to
the longjmp-based scheduler, and the directory usecases/time-series-inlined
provides an (earlier) version that just inlines nodes insteaf of truly
instantiate them.

### Build the use case

Move to the ```time-series``` directory and build the use case ```make```
The result of compilation is file ```rnn```. It prints the timestamp at each
cycle.


