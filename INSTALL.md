The installation of mlir-lus is a rather long process, due to the 
need to compile LLVM/MLIR and IREE:
* Not being an officially-recognized MLIR dialect, and given that
  MLIR rapidly evolves, we set mlir-lus to depend on a specific
  LLVM/MLIR version, which must then be compiled.
* While not being a direct dependency needed in the compilation
  of mlir-lus, IREE and TensorFlow are exposing compilation
  steps that are needed in the compilation of the use cases.
  Again, this requires the use of specific versions (and therefore
  recompilation) to ensure the compatibility between the compilation
  steps that we use together.
  
We are currently in the process of reducing this dependency on 
specific versions of IREE and TensorFlow by trying to reuse full
compilation pipelines they expose (instead of just compilation
steps). 

## Step 1: Install dependences
Our tool is built on top of LLVM/MLIR, which itself requires:
* A recent version of clang (we recommend clang-9). For Ubuntu Linux, it can 
  be installed with:
  ```
  sudo apt-get install clang-9 llvm-9.0 llvm-devel clang-devel
  ```
* A recent version of cmake (>= cmake 13). For Ubuntu Linux, it must be 
  installed from sources to be downloaded at https://cmake.org/download
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

## Step 2: Install LLVM/MLIR
### Step 2.1: Clone the github repository
```
git clone https://github.com/llvm/llvm-project
git checkout bcd6424f9b693af57b29a0f03c52d6991be35d41
```
### Step 2.2: Compilation and installation
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
	
## Step 3: Install IREE and Tensorflow
### Step 3.1: Clone the github repository
```
git clone https://github.com/google/iree.git
git checkout 4f218a8be5ba3e840ebfc8300c9124d01ab0ecc1
git submodule update --init
```
### Step 3.2: Compilation of Tensorflow (as a dependence of IREE)
```
cd iree/third_party/tensorflow
git submodule update --init
bazel build //tensorflow/compiler/mlir:tf-mlir-translate
bazel build //tensorflow/compiler/mlir:tf-opt
```

After compilation, add to the ```$PATH``` environment variable the 
following path:
```
$PATH_TO_IREE/iree/third_party/tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/mlir
```

### Step 3.3: Compilation of IREE
From the iree directory :
```
python3 configure_bazel.py
bazel build //tensorflow/compiler/mlir:tf-mlir-translate
bazel build //iree/tools:iree-opt
```
   
After compilation, add to the ```$PATH``` environment variable the 
following path: 
```
$PATH_TO_IREE/iree/bazel-out/k8-opt/bin/iree/tools
```

## Step 4: Compilation of mlirlus
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

## Step 5: Compilation of mlir-prime

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

## Step 6. Installationn of Sound eXchange (SoX)
This is a sound processing toolbox needed by the Vocoder use case.
* On Ubuntu Linux, it can be installed with ```sudo apt-get install sox```
* Under MacOSX, using ```macports```, the install command 
  is ```sudo port install sox```


# Licensing

This software is released under the GNU General Public License, version 2.0 or later, as detailed in the [LICENSE](LICENSE) file.

