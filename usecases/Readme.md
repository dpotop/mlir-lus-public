# Reactive MLIR dialect - use cases and installation/compilation instructions

We provide in this folder the source code of the three use
cases. For each of the two use cases, we provide two variants
exploring different specification and/or implementation styles.

We also provide compilation instructions allowing the installation
of externally-needed tools such as the ```tensorflow``` and ```iree```
compilers. Note that these tools are currently evolving (rapidly, for
some), and so it is important to use the specific versions indicated
below. Current versions are not guaranteed to work.

## External packages needed for compilation
Our tool is built on top of LLVM/MLIR, which itself requires:
* A recent version of clang (we recommend clang-9). For Ubuntu Linux, it can 
  be installed with:
  ```
  sudo apt-get install clang-9 llvm-9.0 llvm-devel clang-devel python3
  ```
* A recent version of cmake (>= cmake 13). For Ubuntu Linux, it must be 
  installed from sources to be downloaded at ```https://cmake.org/download```
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
- On MacOs over M1 processors:
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
### Compilation of Tensorflow as an iree dependency
From iree directory :
```
cd third_party/tensorflow
git submodule update --init
bazel build //tensorflow/compiler/mlir:tf-mlir-translate
bazel build //tensorflow/compiler/mlir:tf-opt
```
After compilation, 
```
iree/third_party/tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/mlir
```
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
The vocoder use case requires the use of the sox sound processing toolbox. 
* On Ubuntu Linux, it can be installed with ```sudo apt-get install sox```
* Under MacOSX, using ```macports```, the install command 
  is ```sudo port install sox```

## MLIRLus
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

## Usecases

### Pitch tuning vocoder

#### Build the use case

Move to the ```pitch-sched``` directory.

1- If needed, change variables on top of the Makefile. Variable ```LLVMDIR``` 
   must be set to the installation folder of ```llvm``` 
   (normally ```$(HOME)/llvm```). Variable ```MLIRLUS ``` must be set to the 
   path of your build of ```mlir-lus``` (normally ```../mlir-lus```).

2- Build the use case in the pitch-lus-threads directory with ```make```

The result of compilation is file ```pitch```.

#### Running the use case

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

#### Description of the use case

The source code of the use case consists of files of 3 types:
* ```lus``` dialect code is grouped in file ```pitch.lus```, 
  where it is mixed with regular MLIR code of the ```Standard```
  dialect. 

* Regular MLIR code is grouped in the ```*.mlir``` files.
  This comprises the majority of the use case. It can be 
  integrated with the reactive code, but we preferred keeping
  the reactive code apart for lisibility.
  
* C code:
  * TODO scheduler
  * ```sndio.c``` provides the soundcard read and write routines,
    which cannot be written in MLIR in the standard dialect (only
	in the lower-level LLVM dialect).
  * ```bitrev.c``` is the only only function written in C
    that does not interface with the OS. It can be
    converted to MLIR, but we preferred showcasing the ability 
	to interface.

#### The reactive code and its compilation

All reactive code is contained in file ```pitch.lus``` and 
consists of standard MLIR and ```lus``` dialect code.
The code is divided in 3 nodes:
* Node ```@pitch``` implements the top-level I/O and control,
  including the sliding window over input samples, and the
  mute/unmute control. It
  instantiates nodes ```@pitch_algo``` and ```@kbd_ctrl```
* Node ```@pitch_algo``` implements the pitch shifting
  signal processing algorithm.
* Node ```@kbd_ctrl``` implements the keyboard interaction
   algorithm that changes the pitch correction configuration.	
	
NOTE: In the source code, all ```lus``` and ```sync``` dialect 
operations are prefixed with the dialect name. Thus, ```lus.fby``` 
is the operation ```fby``` of dialect ```lus```.

Our reactive source code showcases all features of the Lustre 
language described in the paper:
* Cyclic execution model (implicit)
* Cyclic I/O through the interface variables of the three nodes.
* Feedback (program state) implemented using the ```lus.fby```
  operation.
* Dataflow expression of conditional execution, using the 
  ```lus.when``` and ```lus.merge``` operations.
* Modularity.

The first step in the compilation of this code is to lower
it to the abstraction level of the ```sync``` dialect
described in Section 3 of the submitted paper. This 
is done through the command:

```../mlir-lus/mlirlus --normalize --convert-lus-to-sync pitch.lus```

At this level:
* State is encoded using existing mechanisms provided by standard SSA.
  In our paper, this is done using variables and phi operations. 
  The output of our tool uses higher-level structured control 
  flow constructs provided by MLIR (as part of the ```SCF``` dialect),
  which facilitate subsequent memory allocation phases (but will 
  ultimately be lowered to standard dialect).
* Variable-based inputs and outputs have been transformed into 
  I/O channel variables and ```sync.input``` and ```sync.output``` operations.
* The implicit cyclic execution model, implicit activation conditions,
  and implicit cycle separation is made explicit using standard 
  SSA control flow and the ```sync.tick``` operation. This operation also
  requires the explicit use of ```sync.undef``` operations.

From this level, a further lowering phase, materialized in the 
command line option ```--convert-sync-to-std``` will lower 
the ```sync``` dialect operations to standard dialect MLIR, 
including calls to library functions such as ```tick()```, 
defined below. 

At this point, no memory allocation has yet been performed. The 
compilation script does that using the ```mlir-opt``` tool 
of MLIR, as described in the Makefile.

### Time Series application

TODO - add description

### Resnet50

TODO - add description
TODO - add instructions on importing the example from Keras, or the 
       code with constant weights (and specify it).
