# Set up compilation configuration 
include ../makefile.config

# Application name
APPNAME = rnn
APPNAME_SEQ = $(APPNAME)-seq

TF_FILES = $(APPNAME).tf.mlir

MLIR_MEMREF_FILES = rnn-aux.memref.mlir
ALLMLIR_INLINE =  $(TF_FILES:.mlir=.tensor.memref.mlir) $(MLIR_MEMREF_FILES)
ALLMLIR_SEQ =  $(TF_FILES:.tf.mlir=-inline.tf.tensor.memref.mlir) $(MLIR_MEMREF_FILES) main.memref.mlir

RUNTIME_DIR = ../runtime

CFILES_INLINE = $(RUNTIME_DIR)/scheduler.c $(RUNTIME_DIR)/scheduler_io.c memrefs.c main.c
CFILES_SEQ = tick.c

$(APPNAME): $(ALLMLIR_INLINE:.mlir=.mlir.o) $(CFILES_INLINE:.c=.c.o)
	$(CC) $^ -o $@ $(LDFLAGS)

sequential: $(ALLMLIR_SEQ:.mlir=.mlir.o) $(CFILES_SEQ:.c=.c.o)
	$(CC) $^ -o $(APPNAME_SEQ) $(LDFLAGS)

%.mlir.o: %.mlir.s
	@echo "Compiling $< to $@"
	@$(CC) -c $< -o $@

%.mlir.s: %.mlir.bc
	@echo "Compiling $< to $@"
	@$(LLC) -O3 $< -o=$@

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

%.tf.tensor.mlir : %.tf.mlir
	$(TF_OPT) $< --canonicalize --sccp "-xla-legalize-tf=allow-partial-conversion use-tf2xla-fallback=true device-type=" --canonicalize |\
	$(IREE_OPT) -iree-flow-hlo-to-hlo-preprocessing \
	-iree-flow-extract-pad-from-conv \
	--iree-codegen-hlo-to-linalg-on-tensors \
	--linalg-fold-unit-extent-dims \
	--canonicalize --iree-codegen-fusion-of-tensor-ops \
	--cse --iree-codegen-hlo-to-linalg-on-tensors -o=$@

%.tf.mlir : %.lus.mlir
	@echo "Compiling $< to $@"
	@$(MLIRLUS) $< --normalize --convert-lus-to-sync --convert-sync-to-std -o $@

%-inline.tf.mlir : %.lus.mlir
	@echo "Compiling $< to $@"
	@$(MLIRLUS) $< --normalize --inline-nodes --convert-lus-to-sync --convert-sync-to-std -o $@



%.c.o: %.c
	$(CC) -c $< -o $@

clean:
	rm -f $(APPNAME) $(APPNAME_SEQ) *.o
