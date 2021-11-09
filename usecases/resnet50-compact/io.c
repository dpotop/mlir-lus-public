// These are the low-level routines accessing the sound device
// using the sox utility. They use the POSIX "read" and "write"
// system functions and manipulate C pointers, meaning that
// they cannot be modeled using standard MLIR memrefs. This is
// why they are written in C. They produce output into MLIR-allocated
// fixed-size memrefs.

#include <stdint.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>

//============================================================
// Data structure allowing communicating the samples to
// the MLIR code under the form of a fixed-size memref.
// All memory allocation must be handled by the caller.
typedef struct {
  float *allocd ;
  float *aligned ;
  intptr_t offset ;
  intptr_t size[1] ;
  intptr_t stride[1] ;
} memref_type ;



//
void _mlir_ciface_read_tensors(int32_t pos, memref_type*mt) {
  assert(mt->offset==0);
  assert(mt->allocd==mt->aligned);
}


//============================================================
// The write routine does not need a  specified size like
// SAMPLE_SIZE. It has already been provided in the
// struct.
void _mlir_ciface_write_tensors(int32_t pos, memref_type*mt) {
  assert((mt->offset==0)
	 &&(mt->allocd==mt->aligned));
}
