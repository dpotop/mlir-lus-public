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
  int16_t *allocd ;
  int16_t *aligned ;
  intptr_t offset ;
  intptr_t size[1] ;
  intptr_t stride[1] ;
} memref_type ;



//
void _mlir_ciface_read_samples(int32_t pos, memref_type*mt) {
/* void _mlir_ciface_read_samples(int32_t inst, int32_t pos, memref_type*mt) { */
  // Sanity check
  assert(mt->offset==0);
  assert(mt->stride[0]==1);
  assert(mt->allocd==mt->aligned);
  // Reading the data from standard input
  int r ;
  int r_acc = mt->size[0]*sizeof(int16_t) ;
  char* tmp_buf = (char*)mt->aligned ;
  for(;r_acc;r_acc-=r,tmp_buf+=r) {
    r = read(0,tmp_buf,r_acc) ;
    if(r<0) {
      perror("read_samples error:") ;
      exit(0) ;
    } 
  }
}


//============================================================
// The write routine does not need a  specified size like
// SAMPLE_SIZE. It has already been provided in the
// struct.
int32_t _mlir_ciface_write_samples(int32_t pos, memref_type*mt) {
/* void _mlir_ciface_write_samples(int32_t inst, int32_t pos, memref_type*mt) { */
  // void write_samples(memref_type*mt) {
  // Sanity check
  assert((mt->offset==0)
	 &&(mt->stride[0]==1)
	 &&(mt->allocd==mt->aligned));
  // Write the data
  int w ;
  int w_acc = mt->size[0]*sizeof(int16_t) ;
  char* tmp_buf = (char*)mt->aligned ;
  for(;w_acc;w_acc-=w,tmp_buf+=w) {
    w = write(1,tmp_buf,w_acc) ;
    if(w<0) {
      perror("write_samples error:") ;
      exit(0) ;
    }
  }
  fflush(stdout) ;
  return 0;
}


//============================================================
// The write routine does not need a  specified size like
// SAMPLE_SIZE. It has already been provided in the
// struct.
static int kbd_fd = -1 ;
void open_kbd() {
  kbd_fd = open("kbd",O_RDONLY|O_NONBLOCK) ;
  if(kbd_fd<0){
    perror("Cannot open named fifo.\n") ;
    //    exit(0) ;
  }
}

typedef struct {
  int8_t *allocd ;
  int8_t *aligned ;
  intptr_t offset ;
  intptr_t size[1] ;
  intptr_t stride[1] ;
} memref_i8_type ;

void _mlir_ciface_read_kbd(int32_t pos, memref_i8_type*mt) {
  assert(mt->offset==0);
  assert(mt->stride[0]==1);
  assert(mt->allocd==mt->aligned);
  char buf[1] ;
  if(read(kbd_fd,buf,1)==1)
    *(mt->allocd) =  buf[0] ;
}
