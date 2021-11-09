#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

// This is currently the only function written in C
// that does not interface with the OS. It should be
// converted to MLIR at some point (ideally memref-based,
// to avoid useless copies). But it's not fundamental,
// so we kept it this way, with a tensorial
// interface in main.mlir (the initialization part).
typedef struct {
  int *allocd ;
  int *aligned ;
  intptr_t offset ;
  intptr_t size[1] ;
  intptr_t stride[1] ;
} memref_type_i32 ;
static inline void swap(memref_type_i32*mr,
			unsigned int forward,
                        unsigned int rev) {
  int tmp;
  tmp = mr->aligned[forward];
  mr->aligned[forward] = mr->aligned[rev];
  mr->aligned[rev] = tmp;
}
void _mlir_ciface_bitrev_init(memref_type_i32*mr) {
  int N = mr->size[0] ;
  for(int i=0;i<N;i++)mr->aligned[i] = i ;
  
  const unsigned int halfn = N>>1;
  const unsigned int quartn = N>>2;
  const unsigned int nmin1 = N-1;
  
  // Variables
  unsigned int i, forward, rev, zeros;
  unsigned int nodd, noddrev;
  
  // Variable initialization
  forward = halfn;
  rev = 1;
  
  // start of bitreversed permutation loop, N/4 iterations
  for(i=quartn; i; i--) {
    // Gray code generator for even values:
    nodd = ~i;                                // counting ones is easier
    for(zeros=0; nodd&1; zeros++) nodd >>= 1; // find trailing zeros in i
    forward ^= 2 << zeros;                    // toggle one bit of forward
    rev ^= quartn >> zeros;                   // toggle one bit of rev

    // swap even and ~even conditionally
    if(forward<rev) {
      swap(mr,forward,rev);
      nodd = nmin1 ^ forward;                 // compute the bitwise negations
      noddrev = nmin1 ^ rev;
      swap(mr,nodd,noddrev);                  // swap bitwise-negated pairs
    }
    nodd = forward ^ 1;                       // compute the odd values from the even
    noddrev = rev ^ halfn;
    swap(mr,nodd,noddrev);                    // swap odd unconditionally
  }
}
