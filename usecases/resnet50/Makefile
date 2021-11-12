# Set up compilation configuration 
include ../makefile.config

# Application name
APPNAME = resnet
APPNAME_SEQ = resnet-seq

TF_FILES = $(APPNAME).tf.mlir
MLIR_LUS_FILES = $(APPNAME).lus.mlir
MLIR_MEMREF_FILES = main.memref.mlir

RUNTIME_DIR = ../runtime

CFILES = $(RUNTIME_DIR)/scheduler.c $(RUNTIME_DIR)/scheduler_io.c memrefs.c main.c
CFILES_SEQ = io.c tick.c

ALLMLIR = $(MLIR_LUS_FILES:.mlir=.tf.tensor.memref.mlir) $(TF_FILES:.mlir=.tensor.memref.mlir)
ALLMLIR_SEQ = main.memref.mlir $(MLIR_LUS_FILES:.mlir=.tf.tensor.memref.mlir) $(TF_FILES:.mlir=.tensor.memref.mlir)

$(APPNAME): $(ALLMLIR:.mlir=.mlir.o) $(CFILES:.c=.c.o)
	$(LD) $^ -o $@ $(LDFLAGS)

sequential: $(ALLMLIR_SEQ:.mlir=.mlir.o) $(CFILES_SEQ:.c=.c.o)
	$(LD) $^ -o $(APPNAME_SEQ) $(LDFLAGS)

%.mlir.o: %.mlir.s
	@echo "Compiling $< to $@"
	@$(CC) -c $< -o $@

%.mlir.s: %.mlir.bc
	@echo "Compiling $< to $@"
	@$(LLC) $< -o=$@

%.mlir.bc: %.llvm.mlir
	@echo "Compiling $< to $@"
	@$(MLIRTRANS) --mlir-to-llvmir $< -o=$@

%.memref.llvm.mlir: %.memref.mlir
	@echo "Compiling $< to $@"
	@$(MLIRPRIME) $< --remove-copy-prime --prime-linalg-to-affine \
	--canonicalize --convert-linalg-to-affine-loops --cse \
	--loop-permutation-prime=permutation-map=2,3,4,5,0,1,6 \
	--affine-loop-normalize --loop-invariant-code-motion --lower-affine |\
	$(MLIROPT) --buffer-deallocation --convert-scf-to-std --canonicalize --cse --convert-linalg-to-llvm \
	--convert-complex-to-llvm --convert-std-to-llvm -o=$@

%.tensor.memref.mlir: %.tensor.mlir
	@echo "Compiling $< to $@"
	@$(MLIROPT) $< --canonicalize --cse -sccp \
	--func-bufferize --buffer-results-to-out-params \
	--tensor-constant-bufferize --linalg-bufferize --linalg-detensorize |\
	$(MLIRPRIME) --bufferize-linalg-reshape |\
	$(MLIROPT) --tensor-bufferize --scf-bufferize --std-bufferize \
	--finalizing-bufferize --cse -o=$@

%.lus.tf.mlir: %.lus.mlir $(MLIRLUS)
	@echo "Compiling $< to $@"
	@$(MLIRLUS) $< --normalize --convert-lus-to-sync --convert-sync-to-std -o=$@

%.tf.tensor.mlir : %.tf.mlir
	@echo "Compiling $< to $@"
	@$(TF_OPT) $< --canonicalize --sccp "-xla-legalize-tf=allow-partial-conversion use-tf2xla-fallback=true device-type=" --canonicalize |\
	$(IREE_OPT)  -iree-flow-hlo-to-hlo-preprocessing \
	-iree-flow-extract-pad-from-conv \
	--iree-codegen-hlo-to-linalg-on-tensors \
	--linalg-fold-unit-extent-dims \
	--canonicalize --iree-codegen-fusion-of-tensor-ops \
	--cse --iree-codegen-hlo-to-linalg-on-tensors \
	--iree-linalg-on-tensors-path -o=$@

%.c.o: %.c
	$(CC) -c $< -o $@

clean:
	rm -f $(APPNAME) $(APPNAME_SEQ) *.o
