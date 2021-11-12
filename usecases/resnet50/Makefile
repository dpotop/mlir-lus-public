# Set up compilation configuration 
include ../makefile.config

# Application name
APPNAME = resnet
APPNAME_EXT = $(APPNAME)-ext
APPNAME_COMPACT = $(APPNAME)-compact

TF_FILES_EXT = $(APPNAME_EXT).tf.mlir
MLIR_LUS_FILES_EXT = $(APPNAME_EXT).lus.mlir
MLIR_MEMREF_FILES_EXT = main-ext.memref.mlir

MLIR_MEMREF_FILES_COMPACT = main-compact.memref.mlir
MLIR_LUS_FILES_COMPACT = $(APPNAME_COMPACT).lus.mlir

CFILES = io.c tick.c

ALLMLIRFILES_EXT = $(MLIR_MEMREF_FILES_EXT) $(MLIR_LUS_FILES_EXT:.mlir=.tf.tensor.memref.mlir) $(TF_FILES_EXT:.mlir=.tensor.memref.mlir)

ALLMLIRFILES_COMPACT = $(MLIR_MEMREF_FILES_COMPACT) $(MLIR_LUS_FILES_COMPACT:.mlir=.tf.tensor.memref.mlir)

all: $(APPNAME_EXT) $(APPNAME_COMPACT)

$(APPNAME_EXT): $(ALLMLIRFILES_EXT:.mlir=.mlir.o) $(CFILES:.c=.c.o)
	$(LD) $^ -o $@ $(LDFLAGS)

$(APPNAME_COMPACT): $(ALLMLIRFILES_COMPACT:.mlir=.mlir.o) $(CFILES:.c=.c.o)
	$(LD) $^ -o $@ $(LDFLAGS)

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
	rm -f $(APPNAME)-wrapper.lus.tensor.mlir $(APPNAME)-wrapper.lus.tensor.memref.mlir
	rm -f $(APPNAME)-wrapper.lus.tensor.memref.llvm.mlir
	rm -f $(APPNAME_EXT) $(APPNAME_COMPACT) *.o
