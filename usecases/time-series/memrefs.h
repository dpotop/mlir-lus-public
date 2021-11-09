#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <stdint.h>

#include "error.h"

#ifndef MEMREFS_H
#define MEMREFS_H

struct memref_2d_i32 {
  int32_t* allocated ;
  int32_t* aligned ;
  intptr_t offset ;
  intptr_t size[2] ;
  intptr_t stride[2] ;
};

void alloc_memref_2d_i32(struct memref_2d_i32*s, int size0, int size1);
void free_memref_2d_i32(struct memref_2d_i32*s);
void print_memref_2d_i32(struct memref_2d_i32*s);
void bzero_memref_2d_i32(struct memref_2d_i32*s);

struct memref_2d_f32 {
  float* allocated ;
  float* aligned ;
  intptr_t offset ;
  intptr_t size[2] ;
  intptr_t stride[2] ;
};

void alloc_memref_2d_f32(struct memref_2d_f32*s, int size0, int size1);
void free_memref_2d_f32(struct memref_2d_f32*s);
void print_memref_2d_f32(struct memref_2d_f32*s);
void bzero_memref_2d_f32(struct memref_2d_f32*s);

struct memref_1d_i16 {
  int16_t* allocated ;
  int16_t* aligned ;
  intptr_t offset ;
  intptr_t size[1] ;
  intptr_t stride[1] ;
};

void alloc_memref_1d_i16(struct memref_1d_i16*s, int size0);
void free_memref_1d_i16(struct memref_1d_i16*s);
void print_memref_1d_i16(struct memref_1d_i16*s);
void bzero_memref_1d_i16(struct memref_1d_i16*s);

struct memref_1d_i8 {
  int8_t* allocated ;
  int8_t* aligned ;
  intptr_t offset ;
  intptr_t size[1] ;
  intptr_t stride[1] ;
};

void alloc_memref_1d_i8(struct memref_1d_i8*s, int size0);
void free_memref_1d_i8(struct memref_1d_i8*s);
void print_memref_1d_i8(struct memref_1d_i8*s);
void bzero_memref_1d_i8(struct memref_1d_i8*s);


#endif
