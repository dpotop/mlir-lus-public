func @undef_tensor_3_100_f32(%r:memref<3x100xf32>) {
  return
}

func @undef_tensor_3_1_f32(%r:memref<3x1xf32>) {
  return
}

func @undef_tensor_3_50_f32(%r:memref<3x50xf32>) {
  return
}

func @undef_tensor_50_f32(%r:memref<50xf32>) {
  return
}

func @undef_tensor_100_50_f32(%r:memref<100x50xf32>) {
  return
}

func @undef_tensor_50_50_f32(%r:memref<50x50xf32>) {
  return
}

func @undef_tensor_1_f32(%r:memref<1xf32>) {
  return
}

func @undef_tensor_50_1_f32(%r:memref<50x1xf32>) {
  return
}

func @undef_i32() -> (i32) {
  %c = constant 0: i32
  return %c: i32
}

func @select_tensor_3_100_f32(%c:i1,%t:memref<3x100xf32>,%f:memref<3x100xf32>,%r:memref<3x100xf32>) {
  %to_copy = select %c,%t,%f : memref<3x100xf32>
  linalg.copy(%to_copy,%r) : memref<3x100xf32>,memref<3x100xf32>
  return
}
