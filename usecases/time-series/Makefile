LLVMDIR=$(HOME)/llvm-old
MLIROPT = $(LLVMDIR)/bin/mlir-opt
MLIRTRANS = $(LLVMDIR)/bin/mlir-translate
LLC = llc

MLIRLUSDIR=../../mlir-lus
MLIRLUS = $(MLIRLUSDIR)/mlirlus

MLIRPRIMEDIR=../../mlir-prime
MLIRPRIME = $(MLIRPRIMEDIR)/mlir-prime

IREE_BIN_DIR=/home/hpompougnac/.cache/bazel/_bazel_hpompougnac/c21dee9b578a4b1c3f60a275119db250/execroot/iree_core/bazel-out/k8-opt/bin
TF_BIN_DIR=/home/hpompougnac/.cache/bazel/_bazel_hpompougnac/52c22a949d9769dd1d18ce3d575a5e05/execroot/org_tensorflow/bazel-out/k8-opt/bin
IREE_OPT=$(IREE_BIN_DIR)/iree/tools/iree-opt
TF_OPT=$(TF_BIN_DIR)/tensorflow/compiler/mlir/tf-opt

CC=clang
LDFLAGS=-lm

APPNAME = rnn

TF_FILES = $(APPNAME).tf.mlir

# MLIR_LUS_FILES = $(APPNAME)-wrapper.lus.mlir

MLIR_MEMREF_FILES = rnn-aux.memref.mlir

CFILES = scheduler.c scheduler-aux.c memrefs.c main.c 

# ALLMLIRFILES = $(TF_FILES:.mlir=.tensor.memref.mlir) $(MLIR_MEMREF_FILES) $(MLIR_LUS_FILES:.mlir=.tensor.memref.mlir)

ALLMLIRFILES =  $(TF_FILES:.mlir=.tensor.memref.mlir) $(MLIR_MEMREF_FILES)

all: $(APPNAME)

$(APPNAME): $(ALLMLIRFILES:.mlir=.mlir.o) $(CFILES:.c=.c.o)
	$(CC) $^ -o $@ $(LDFLAGS)

%.mlir.o: %.mlir.s
	@echo "Compiling $< to $@"
	@$(CC) -c $< -o $@

%.mlir.s: %.mlir.bc
	@echo "Compiling $< to $@"
	@$(LLC) -O3 --fp-contract=fast $< -o=$@

%.mlir.bc: %.llvm.mlir
	@echo "Compiling $< to $@"
	@$(MLIRTRANS) --mlir-to-llvmir $< -o=$@

%.memref.llvm.mlir: %.memref.mlir
	@echo "Compiling $< to $@"
	@$(MLIRPRIME) $< --remove-copy-prime --prime-linalg-to-affine \
	--canonicalize --convert-linalg-to-affine-loops --cse \
	--loop-permutation-prime=permutation-map=2,3,4,5,0,1,6 \
	--affine-loop-normalize --lower-affine |\
	$(MLIROPT) --convert-scf-to-std --test-math-polynomial-approximation \
	--convert-linalg-to-llvm  --convert-complex-to-llvm --convert-std-to-llvm -o=$@

%.tensor.memref.mlir: %.tensor.mlir
	@echo "Compiling $< to $@"
	@$(MLIROPT) $< --canonicalize --cse \
	--func-bufferize --buffer-results-to-out-params \
	--tensor-constant-bufferize --linalg-bufferize --linalg-detensorize |\
	$(MLIRPRIME) --bufferize-linalg-reshape |\
	$(MLIROPT) --tensor-bufferize --scf-bufferize --std-bufferize \
	 --buffer-deallocation --finalizing-bufferize --cse -o=$@

%.tf.tensor.mlir : %.lus.mlir
	@echo "Compiling $< to $@"
	@$(MLIRLUS) $< --normalize --convert-lus-to-sync --convert-sync-to-std |\
	$(TF_OPT) --canonicalize --sccp "-xla-legalize-tf=allow-partial-conversion use-tf2xla-fallback=true device-type=" --canonicalize |\
	$(IREE_OPT) -iree-flow-hlo-to-hlo-preprocessing \
	-iree-flow-extract-pad-from-conv \
	--iree-codegen-hlo-to-linalg-on-tensors \
	--linalg-fold-unit-extent-dims \
	--canonicalize --iree-codegen-fusion-of-tensor-ops \
	--cse --iree-codegen-hlo-to-linalg-on-tensors -o=$@


%.c.o: %.c
	$(CC) -c $< -o $@

clean:
	rm -f $(APPNAME)-wrapper.lus.tensor.mlir $(APPNAME)-wrapper.lus.tensor.memref.mlir
	rm -f $(APPNAME)-wrapper.lus.tensor.memref.llvm.mlir
	rm -f $(APPNAME) *.o
