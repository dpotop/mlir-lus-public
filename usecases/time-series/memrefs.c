#include "memrefs.h"

// struct memref_2d_i32

void alloc_memref_2d_i32(struct memref_2d_i32*s,
			 int size0, int size1) {
  int size = size0*size1*sizeof(int32_t) ;
  // For some reason, aligned_alloc always returns NULL.
  // This is why I use an malloc-based approach.
  // s->allocated = aligned_alloc(sizeof(int32_t),size) ;
  s->allocated = malloc(size+sizeof(int32_t)) ;
  uintptr_t offset = ((uintptr_t)s->allocated)%((uintptr_t)sizeof(int32_t)) ;
  if(offset>0) {
    offset = ((uintptr_t)sizeof(int32_t))-offset ;
    s->allocated = (void*)(((uintptr_t)s->allocated)+offset) ;
  }
  s->aligned = s->allocated ;
  s->offset = 0 ;
  s->size[0] = size0 ;
  s->size[1] = size1 ;
  s->stride[0] = 1 ;
  s->stride[1] = 1 ;
}

void free_memref_2d_i32(struct memref_2d_i32*s) {
  free(s->allocated) ;
}

void print_memref_2d_i32(struct memref_2d_i32*s) {
  printf("memref<%ldx%ldxi32>:\n",s->size[0],s->size[1]) ;
  for(int i=0;i<s->size[0];i++) {
    printf("\t") ;
    for(int j=0;j<s->size[1];j++)
      printf("%d ",s->aligned[i*s->size[1]+j]) ;
    printf("\n") ;
  }
  fflush(stdout) ;
}

void bzero_memref_2d_i32(struct memref_2d_i32*s) {
  bzero(s->aligned,s->size[0]*s->size[1]*sizeof(int32_t)) ;
}

// struct memref_2d_f32

void alloc_memref_2d_f32(struct memref_2d_f32*s,
			 int size0, int size1) {
  int size = size0*size1*sizeof(float) ;
  // For some reason, aligned_alloc always returns NULL.
  // This is why I use an malloc-based approach.
  // s->allocated = aligned_alloc(sizeof(float),size) ;
  s->allocated = malloc(size+sizeof(float)) ;
  uintptr_t offset = ((uintptr_t)s->allocated)%((uintptr_t)sizeof(float)) ;
  if(offset>0) {
    offset = ((uintptr_t)sizeof(float))-offset ;
    s->allocated = (void*)(((uintptr_t)s->allocated)+offset) ;
  }
  s->aligned = s->allocated ;
  s->offset = 0 ;
  s->size[0] = size0 ;
  s->size[1] = size1 ;
  s->stride[0] = 1 ;
  s->stride[1] = 1 ;
}

void free_memref_2d_f32(struct memref_2d_f32*s) {
  free(s->allocated) ;
}

void print_memref_2d_f32(struct memref_2d_f32*s) {
  printf("memref<%ldx%ldxf32>:\n",s->size[0],s->size[1]) ;
  for(int i=0;i<s->size[0];i++) {
    printf("\t") ;
    for(int j=0;j<s->size[1];j++)
      printf("%f ",s->aligned[i*s->size[1]+j]) ;
    printf("\n") ;
  }
  fflush(stdout) ;
}

void bzero_memref_2d_f32(struct memref_2d_f32*s) {
  bzero(s->aligned,s->size[0]*s->size[1]*sizeof(float)) ;
}

// struct memref_1d_i16

void alloc_memref_1d_i16(struct memref_1d_i16*s,
			 int size0) {
  int size = size0*sizeof(int16_t) ;
  // For some reason, aligned_alloc always returns NULL.
  // This is why I use an malloc-based approach.
  // s->allocated = aligned_alloc(sizeof(int16_t),size) ;
  s->allocated = malloc(size+sizeof(int16_t)) ;
  uintptr_t offset = ((uintptr_t)s->allocated)%((uintptr_t)sizeof(int16_t)) ;
  if(offset>0) {
    offset = ((uintptr_t)sizeof(int16_t))-offset ;
    s->allocated = (void*)(((uintptr_t)s->allocated)+offset) ;
  }
  s->aligned = s->allocated ;
  s->offset = 0 ;
  s->size[0] = size0 ;
  s->stride[0] = 1 ;
}

void free_memref_1d_i16(struct memref_1d_i16*s) {
  free(s->allocated) ;
}

void print_memref_1d_i16(struct memref_1d_i16*s) {
  printf("memref<%ldxi16>:\n",s->size[0]) ;
  for(int i=0;i<s->size[0];i++) {
    printf("%d ", s->aligned[i]);
  }
  fflush(stdout) ;
}

// struct memref_1d_i8

void alloc_memref_1d_i8(struct memref_1d_i8*s,
			 int size0) {
  int size = size0*sizeof(int8_t) ;
  // For some reason, aligned_alloc always returns NULL.
  // This is why I use an malloc-based approach.
  // s->allocated = aligned_alloc(sizeof(int8_t),size) ;
  s->allocated = malloc(size+sizeof(int8_t)) ;
  uintptr_t offset = ((uintptr_t)s->allocated)%((uintptr_t)sizeof(int8_t)) ;
  if(offset>0) {
    offset = ((uintptr_t)sizeof(int8_t))-offset ;
    s->allocated = (void*)(((uintptr_t)s->allocated)+offset) ;
  }
  s->aligned = s->allocated ;
  s->offset = 0 ;
  s->size[0] = size0 ;
  s->stride[0] = 1 ;
}

void free_memref_1d_i8(struct memref_1d_i8*s) {
  free(s->allocated) ;
}

void print_memref_1d_i8(struct memref_1d_i8*s) {
  printf("memref<%ldxi8>:\n",s->size[0]) ;
  for(int i=0;i<s->size[0];i++) {
    printf("%d ", s->aligned[i]);
  }
  fflush(stdout) ;
}
