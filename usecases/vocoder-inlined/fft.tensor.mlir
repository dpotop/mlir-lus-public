// This is the part of the MLIR algorithms that can be:
// - represented and compiled using tensors
// - fully bufferized under the hypothesis that
//   tensor outputs are passed on through input memrefs
//   after bufferization
func private @complex_add(complex<f32>,complex<f32>)->complex<f32>
func private @complex_sub(complex<f32>,complex<f32>)->complex<f32>
func private @complex_mul(complex<f32>,complex<f32>)->complex<f32>

// Bit reverse protocol
func @bitrev(%perm:tensor<?xi32>,
             %bf_res:tensor<?xcomplex<f32>>) -> tensor<?xcomplex<f32>> {
  %0 = constant 0 : index
  %size = memref.dim %bf_res, %0 : tensor<?xcomplex<f32>>
  %o = tensor.generate %size {
  ^bb0(%i : index):
    %idxi = tensor.extract %perm[%i]: tensor<?xi32>
    %idx = std.index_cast %idxi : i32 to index
    %elt = tensor.extract %bf_res[%idx] : tensor<?xcomplex<f32>>
    tensor.yield %elt:complex<f32>
  } : tensor<?xcomplex<f32>>
  return %o:tensor<?xcomplex<f32>>
}

// The following two functions should be unified, if only I had a
// map with multiple outputs
func @bf1_up(%tw:complex<f32>,%i0:complex<f32>,%i1:complex<f32>)->complex<f32> {
  %o0 = call @complex_add(%i0,%i1):(complex<f32>,complex<f32>)->complex<f32>
  return %o0:complex<f32>
}
func @bf1_down(%tw:complex<f32>,%i0:complex<f32>,%i1:complex<f32>)->complex<f32> {
  %x = call @complex_sub(%i0,%i1):(complex<f32>,complex<f32>)->complex<f32>
  %o1 = call @complex_mul(%x,%tw):(complex<f32>,complex<f32>)->complex<f32>
  return %o1:complex<f32>
}
