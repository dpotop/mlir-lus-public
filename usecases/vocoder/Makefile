# Set up compilation configuration 
include ../makefile.config

# Application name
APPNAME = pitch

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

RUNTIME_DIR = ../runtime

CFILES = \
	bitrev.c \
	main.c \
	memrefs.c \
	$(RUNTIME_DIR)/scheduler.c \
	$(RUNTIME_DIR)/scheduler_io.c \

#-------------------------------------------------------------
# Rules. DO NOT CHANGE
#-------------------------------------------------------------
ALLMLIRFILES = $(MLIR_MEMREF_FILES) $(MLIR_TENSOR_FILES:.mlir=.memref.mlir) $(MLIR_LUS_FILES:.mlir=.tensor.memref.mlir)  

# all: pitch.lus.tensor.mlir
all: $(APPNAME)

.SECONDARY: pitch.lus.tensor.mlir pitch.lus.tensor.memref.mlir pitch.lus.tensor.memref.llvm.mlir

$(APPNAME): $(ALLMLIRFILES:.mlir=.mlir.o) $(CFILES:.c=.c.o)
	$(LD) $^ -o $@ $(LDFLAGS)

%.mlir.o: %.mlir.s
	$(CC) -c $< -o $@

%.mlir.s: %.mlir.bc
	$(LLC) $< -o=$@

%.mlir.bc: %.llvm.mlir
	$(MLIRTRANS) --mlir-to-llvmir $< -o=$@

%.memref.llvm.mlir: %.memref.mlir
	$(MLIROPT) --convert-linalg-to-affine-loops \
			--lower-affine --convert-scf-to-std \
			--convert-complex-to-llvm --convert-std-to-llvm $< -o=$@

%.tensor.memref.mlir: %.tensor.mlir
	$(MLIROPT) --tensor-bufferize --scf-bufferize --std-bufferize \
		--func-bufferize --buffer-results-to-out-params \
		--finalizing-bufferize --buffer-deallocation $< -o=$@

%.lus.tensor.mlir: %.lus.mlir $(MLIRLUS)
	$(MLIRLUS) --normalize \
			--convert-lus-to-sync --convert-sync-to-std $< |\
		$(MLIROPT) --canonicalize --cse -sccp -o=$@

%.c.o: %.c
	$(CC) -c $< -o $@

clean:
	rm -f pitch.lus.tensor.mlir pitch.lus.tensor.memref.mlir
	rm -f pitch.lus.tensor.memref.llvm.mlir
	rm -f $(APPNAME) *.o

run-debug: $(DEBUG)
	rec -q -t raw -r 44100 -e signed -b 16 -c 2 - | \
		nice -20 ./$(DEBUG) | \
		play -q -t raw -r 44100 -e signed -b 16 -c 2 -

run: pitch
	mkfifo kbd
	(rec -q -t raw -r 44100 -e signed -b 16 -c 2 - | \
		nice -20 ./$(APPNAME) | \
		play -q -t raw -r 44100 -e signed -b 16 -c 2 - 2>/dev/null&)
	cat >kbd

kill:
	rm -f kbd
	killall -9 pitch
