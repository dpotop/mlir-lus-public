// Implementation of basic operations on complex numbers.

func @complex_add(%i1:complex<f32>,%i2:complex<f32>)->(complex<f32>) {
  %re1 = complex.re %i1 : complex<f32>
  %im1 = complex.im %i1 : complex<f32>
  %re2 = complex.re %i2 : complex<f32>
  %im2 = complex.im %i2 : complex<f32>
  %re = addf %re1, %re2 : f32
  %im = addf %im1, %im2 : f32
  %o = complex.create %re, %im : complex<f32>
  return %o : complex<f32>
}
func @complex_sub(%i1:complex<f32>,%i2:complex<f32>)->(complex<f32>) {
  %re1 = complex.re %i1 : complex<f32>
  %im1 = complex.im %i1 : complex<f32>
  %re2 = complex.re %i2 : complex<f32>
  %im2 = complex.im %i2 : complex<f32>
  %re = subf %re1, %re2 : f32
  %im = subf %im1, %im2 : f32
  %o = complex.create %re, %im : complex<f32>
  return %o : complex<f32>
}
func @complex_mul(%i1:complex<f32>,%i2:complex<f32>)->(complex<f32>) {
  %re1 = complex.re %i1 : complex<f32>
  %im1 = complex.im %i1 : complex<f32>
  %re2 = complex.re %i2 : complex<f32>
  %im2 = complex.im %i2 : complex<f32>
  %x1 = mulf %re1, %re2 : f32 
  %x2 = mulf %im1, %im2 : f32 
  %re = subf %x1, %x2 : f32
  %x3 = mulf %re1, %im2 : f32 
  %x4 = mulf %im1, %re2 : f32 
  %im = addf %x3, %x4 : f32
  %o = complex.create %re, %im : complex<f32>
  return %o : complex<f32>
}
func @complex_swap(%i:complex<f32>)->(complex<f32>) {
  %re = complex.re %i : complex<f32>
  %im = complex.im %i : complex<f32>
  %o = complex.create %im, %re : complex<f32>
  return %o : complex<f32>
}
func @complex_fprod(%i:complex<f32>,%f:f32)->(complex<f32>) {
  %re = complex.re %i : complex<f32>
  %im = complex.im %i : complex<f32>
  %re1 = mulf %re, %f : f32
  %im1 = mulf %im, %f : f32
  %o = complex.create %re1, %im1 : complex<f32>
  return %o : complex<f32>
}

func private @atan2f(f32,f32)->(f32)
func @complex2polar(%i:complex<f32>)->(f32,f32) {
  %re = complex.re %i : complex<f32>
  %im = complex.im %i : complex<f32>
  %x1 = mulf %re, %re : f32 
  %x2 = mulf %im, %im : f32
  %x3 = addf %x1, %x2 : f32
  %mag = math.sqrt %x3 : f32
  %ang = call @atan2f(%im,%re) : (f32,f32)->f32
  return %mag,%ang:f32,f32
}
func @polar2complex(%mag:f32,%ang:f32)->complex<f32>{
  %x1 = math.cos %ang : f32
  %x2 = math.sin %ang : f32
  %re = mulf %mag, %x1 : f32
  %im = mulf %mag, %x2 : f32
  %o = complex.create %re, %im : complex<f32>
  return %o : complex<f32>
}

func @float2complex(%i:f32)->(complex<f32>) {
  %0 = constant 0.0 : f32
  %o = complex.create %i, %0 : complex<f32>
  return %o : complex<f32>
}
func @complex2float(%c:complex<f32>)->(f32,f32) {
  %re = complex.re %c : complex<f32>
  %im = complex.im %c : complex<f32>
  return %re,%im:f32,f32
}

func @complex_tensor_float_product(%data:tensor<?xcomplex<f32>>,%f:f32)->(tensor<?xcomplex<f32>>) {
  %0 = constant 0 : index
  %size = memref.dim %data, %0 : tensor<?xcomplex<f32>>
  %res = tensor.generate %size {
  ^bb0(%idx : index):
    %elt = tensor.extract %data[%idx] : tensor<?xcomplex<f32>>
    %r = call @complex_fprod(%elt,%f) : (complex<f32>,f32)->complex<f32>
    tensor.yield %r:complex<f32>
  } : tensor<?xcomplex<f32>>
  return %res:tensor<?xcomplex<f32>>
}

