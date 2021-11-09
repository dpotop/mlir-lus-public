#### Table of Contents  
* [Installation instructions](#install)
* [Running the use case](#running)
* [Description of the use case](#description)
  * [The reactive code and its compilation](#reactive)
  * [The tick function](#tick)


# Installation instructions <a name="install"/> 

## Step 1: Prerequisites
### External packages
Our tool is built on top of LLVM/MLIR, which itself requires:
* A recent version of clang (we recommend clang-9). For Ubuntu Linux, it can be 
  installed with:
  * ```sudo apt-get install clang-9 llvm-9.0 llvm-devel clang-devel```
* A recent version of cmake (>= cmake 13). For Ubuntu Linux, it must be 
  installed from sources to be downloaded at ```https://cmake.org/download```
* Recent versions of libgmp and z3 (can be found on opam)

On MacOSX using macports, in some cases, full macports removal and 
reinstallation may be needed. This can be done following the instructions
at: https://trac.macports.org/wiki/Migration

### LLVM/MLIR
MLIR is a part of LLVM (it is downloaded and compiled as part of LLVM),
available as such at:
* https://github.com/llvm/llvm-project.git

There are two ways of obtaining this code:
* as a submodule of this project
* as a standalone project
If you get it as a sub-module, it is ensured that the correct
version is downloaded. The disadvantage is that you will have one
full copy of llvm-project inside the current project, whereas
a standalone copy is more easily shared.

#### Download LLVM/MLIR as a submodule
In the root of this project, execute:
```
   git submodule init
   git submodule update
```

Later, to get a new version of LLVM/MLIR:
```
   git submodule update --recursive --remote
```

Full git submodule documentation:
* https://git-scm.com/book/en/v2/Git-Tools-Submodules
* http://openmetric.org/til/programming/git-pull-with-submodule/

#### Download as a standalone repository
From the file listing at https://github.com/dpotop/mlir-rt note the revision of the 
LLVM/MLIR used by our project. We will denote it by XXXX. For instance, if the
file listing lists ```llvm-project @ ea475c7```, then XXXX=ea475c7 .

Then:

1- Pull the LLVM source with ```git clone git@github.com:llvm/llvm-project.git```

2- Choose the right version of LLVM/MLIR, which ensures that our code compiles correctly in the current version:

```
cd llvm-project
git checkout XXXX
```

#### Compilation and installation
We make the assumption that installation prefix is $(HOME)/llvm .

Move to the llvm-project folder and execute. The COMPILER define
directives of the cmake call may have to be updated depending
on the compiler used.
```
cd llvm-project
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_INSTALL_PREFIX=$HOME/llvm -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang-mp-9.0 -DCMAKE_CXX_COMPILER=clang++-mp-9.0 -DCMAKE_ASM_COMPILER=clang-mp-9.0 -DCMAKE_CXX_FLAGS_DEBUG="-fno-omit-frame-pointer -fsanitize=address" -DCMAKE_SHARED_LINKER_FLAGS="-fno-omit-frame-pointer -fsanitize=address" -DCMAKE_EXE_LINKER_FLAGS="-fno-omit-frame-pointer -fsanitize=address " ../llvm
make -j6
make install
```
Attention, depending on the platform, clang-9 and clang++-9 must be replaced with the appropriate compiler names).
The flag -DCMAKE_INSTALL_PREFIX can be changed and tells where LLVM and MLIR will be installed. The compilation flags are set for a debug configuration with run-time memory error detection (which is very useful, given the use of C++ and of complex program rewriting logic).

After compilation, ```$HOME/llvm/bin``` must be added to $PATH.

### Sound eXchange (SoX)
Our use case requires the use of the sox sound processing toolbox. 
* On Ubuntu Linux, it can be installed with ```sudo apt-get install sox```
* Under MacOSX, using ```macports```, the install command is ```sudo port install sox```

## Step 2: Install ```mlirlus``` and the use case

### Step 2.a. Download (PLDI2020 reviewers only)
Along with this file, download the encrypted archive ```pldi2020-419.tar.gz.enc```.

Decrypt it using the password provided along with the download link, using the following command (where XXXXXXXX is replaced by the actual password provided as part of our submission).
```
openssl enc -aes-256-cbc -d -pass pass:XXXXXXXX -in pldi2020-419.tar.gz.enc | tar xz
```
The resulting folder contains:
* the sub-folder "mlir-lus" containing the MLIR extension we propose
* the sub-folder "vocoder-use-case" containing the pitch vocoder 

### Step 2.b: Compile mlirlus

In the mlir-lus folder :

1- Create the file Makefile.config and set within it the variables LLVM, CPP, LDFLAGS, CC. A standard one for Ubuntu can be this one :
```
LLVM=$(HOME)/llvm
CPP=clang++-9
LDFLAGS= -L $(LLVM)/lib -lpthread
CC=clang-9
```

2- Build dependencies with ```touch makefile.depend```, followed by ```make depend```

3- Build mlirlus with ```make```

## Step 3: Build the use case

Move to the ```vocoder-use-case``` directory.

1- If needed, change variables on top of the Makefile. Variable ```LLVMDIR``` must be set to the installation folder of ```llvm``` (normally ```$(HOME)/llvm```). Variable ```MLIRLUS ``` must be set to the path of your build of ```mlir-lus``` (normally ```../mlir-lus```).

2- Build the use case in the pitch-lus-threads directory with ```make```

The result of compilation is file ```pitch```.

# Running the use case <a name="running"/> 

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

# Description of the use case <a name="description"/> 
The source code of the use case consists of files of 3 types:
* ```lus``` dialect code is grouped in file ```pitch.lus```, 
  where it is mixed with regular MLIR code of the ```Standard```
  dialect. 

* Regular MLIR code is grouped in the ```*.mlir``` files.
  This comprises the majority of the use case. It can be 
  integrated with the reactive code, but we preferred keeping
  the reactive code apart for lisibility.
  
* C code:
  * ```tick.c``` provides the implementation of the ```tick```
    function (description below)
  * ```sndio.c``` provides the soundcard read and write routines,
    which cannot be written in MLIR in the standard dialect (only
	in the lower-level LLVM dialect).
  * ```bitrev.c``` is the only only function written in C
    that does not interface with the OS. It can be
    converted to MLIR, but we preferred showcasing the ability 
	to interface.
	
## The reactive code and its compilation <a name="reactive"/> 
	
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
  In particular:
  * All the
  signal processing pipeline of node ```@pitch_algo``` is not
  executed when variable ```%sndon``` is ```false```, as enforced
  by the sub-sampling of the input of ```@pitch_algo``` by the
  ```lus.when``` operations in lines 62 and 63.
  * When this happens, sound output is flat (constant 0), as
    specified using the ```lus.merge``` operation in line 
	78. The result is muted sound output.
* Modularity. In our use case, for simplicity, it is handled 
  at the dataflow level through inlining.

The first step in the compilation of this code is to lower
it to the abstraction level of the ```sync``` dialect
described in Section 3 of the submitted paper. This 
is done through the command:

```../mlir-lus/mlirlus --inline-nodes -mainnode=pitch --normalize --convert-lus-to-sync pitch.lus```

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
of MLIR, as described in the Makefile (in the
rule of lines 79-91).
	
## The ```tick```function <a name="tick"/> 
This function, provided in file ```tick.c``` 
is a soft real-time implementation of the tick operation defined
in the paper).  The objective is here simply to reduce the pressure on
soundcard I/O polling by making the software wait on time for the
largest part of the computation cycle.

To determine the wait duration, one must consider:
* the sampling frequency (44.1kHz = 44100 samples/second)
* the number of samples treated during one execution
  cycle (256 samples/cycle)

These two determine that the computation throughput is
of 44100/256 = 172 cycles/second. This throughput is
naturally ensured by the interaction between the application,
the OS, and the sound HW.

The (average) duration of one execution cycle is therefore
1/172 seconds = 5.81 ms. However, the software only spends
a fraction of this time doing computations. Most of the
time it will wait on the sound input, which potentially involves
continuous polling.

Our objective is to reduce the amount of polling by ensuring
that, for much of the execution cycle, the application simply
waits on time. We do this by using the usleep system service.

The argument provided to usleep must always be smaller than
the cycle average duration of 5.81 ms. Otherwise, the system
is non-schedulable.

Given the complexity of the OSs on which this application
is executed, no good estimation method exists beyond testing.
On a Intel Core i7 quad-core running at 2.5GHz in 16Go of RAM
under MacOSX 10.15.3, 4ms is sometimes insufficient (depending on
the system load), and 3.5ms is a safe value. The value must be
tuned for the architecture. We set here a very safe value of
2ms that should allow execution on most modern platforms.

Note that input to usleep is given in microseconds. Thus, to
specify a delay of 2 ms, one should give usleep 2000 as argument.
Also note that the argument of usleep is a minimum delay.
