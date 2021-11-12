# Set up compilation configuration 
include ../makefile.config

# Application name
APPNAME = pitch
APPNAME_SEQ = pitch-seq

# Highest abstraction level.
# Synchronous and value semantics only.
# Lowering to standard SSA and bufferization needed.
MLIR_LUS_FILES = \
	$(APPNAME).lus.mlir \

# Value semantics only (tensors, no memrefs).
# Bufferization needed.
MLIR_TENSOR_FILES = \
	complex.tensor.mlir \
	fft.tensor.mlir \
	pitch.tensor.mlir \

# No bufferization needed.
MLIR_MEMREF_FILES = \
	fft.memref.mlir \
	pitch.memref.mlir \
	sndio.memref.mlir \

ALLMLIRFILES = $(MLIR_MEMREF_FILES) $(MLIR_TENSOR_FILES:.mlir=.memref.mlir) $(MLIR_LUS_FILES:.mlir=.tensor.memref.mlir)
ALLMLIRFILES_SEQ = main.memref.mlir $(MLIR_MEMREF_FILES) $(MLIR_TENSOR_FILES:.mlir=.memref.mlir) $(MLIR_LUS_FILES:.lus.mlir=-inline.lus.tensor.memref.mlir)

RUNTIME_DIR = ../runtime

CFILES = \
	bitrev.c \
	main.c \
	memrefs.c \
	$(RUNTIME_DIR)/scheduler.c \
	$(RUNTIME_DIR)/scheduler_io.c \

CFILES_SEQ = \
	bitrev.c \
	sndio.c \
	tick.c \

#-------------------------------------------------------------
# Rules. DO NOT CHANGE
#-------------------------------------------------------------

$(APPNAME): $(ALLMLIRFILES:.mlir=.mlir.o) $(CFILES:.c=.c.o)
	$(LD) $^ -o $@ $(LDFLAGS)

sequential: $(ALLMLIRFILES_SEQ:.mlir=.mlir.o) $(CFILES_SEQ:.c=.c.o)
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
	@$(MLIROPT) --convert-linalg-to-affine-loops \
			--lower-affine --convert-scf-to-std \
			--convert-complex-to-llvm --convert-std-to-llvm $< -o=$@

%.tensor.memref.mlir: %.tensor.mlir
	@echo "Compiling $< to $@"
	@$(MLIROPT) --tensor-bufferize --scf-bufferize --std-bufferize \
		--func-bufferize --buffer-results-to-out-params \
		--finalizing-bufferize --buffer-deallocation $< -o=$@

%.lus.tensor.mlir: %.lus.mlir $(MLIRLUS)
	@echo "Compiling $< to $@"
	@$(MLIRLUS) --normalize \
			--convert-lus-to-sync --convert-sync-to-std $< |\
		$(MLIROPT) --canonicalize --cse -sccp -o=$@

%-inline.lus.tensor.mlir: %.lus.mlir $(MLIRLUS)
	@echo "Compiling $< to $@"
	@$(MLIRLUS) --normalize --inline-nodes \
			--convert-lus-to-sync --convert-sync-to-std $< |\
		$(MLIROPT) --canonicalize --cse -sccp -o=$@

%.c.o: %.c
	$(CC) -c $< -o $@

clean:
	rm -f $(APPNAME) $(APPNAME_SEQ) *.o

run-debug: $(DEBUG)
	rec -q -t raw -r 44100 -e signed -b 16 -c 2 - | \
		nice -20 ./$(DEBUG) | \
		play -q -t raw -r 44100 -e signed -b 16 -c 2 -

run: $(APPNAME)
	mkfifo kbd
	(rec -q -t raw -r 44100 -e signed -b 16 -c 2 - | \
		nice -20 ./$< | \
		play -q -t raw -r 44100 -e signed -b 16 -c 2 - 2>/dev/null&)
	cat >kbd

run-seq: $(APPNAME_SEQ)
	mkfifo kbd
	(rec -q -t raw -r 44100 -e signed -b 16 -c 2 - | \
		nice -20 ./$< | \
		play -q -t raw -r 44100 -e signed -b 16 -c 2 - 2>/dev/null&)
	cat >kbd

kill:
	rm -f kbd
	killall -9 pitch
