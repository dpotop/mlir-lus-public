// Memref-based operations used directly by the pitch
// shifting algorithm.

func @concat_aux(%sample_size:index,%offset:index,%circ:memref<256xi16>,%out:memref<1024xf32>) {
  %0 = constant 0 : index
  %1 = constant 1 : index
  scf.for %idx = %0 to %sample_size step %1 {
    %x = memref.load %circ[%idx] : memref<256xi16>
    %y = std.sitofp %x : i16 to f32
    %adr = addi %idx,%offset : index
    memref.store %y, %out[%adr] : memref<1024xf32>
  }
  return
}
func @concat_samples(%circ0:memref<256xi16>,%circ1:memref<256xi16>, // Data input
	    %circ2:memref<256xi16>,%circ3:memref<256xi16>, // Data input
	    %out:memref<1024xf32> // Data output
	    ) {
  %0 = constant 0 : index
  %sample_size = memref.dim %circ0,%0 : memref<256xi16>
  %off0 = constant 0 : index
  %off1 = addi %off0, %sample_size : index
  %off2 = addi %off1, %sample_size : index
  %off3 = addi %off2, %sample_size : index
  call @concat_aux(%sample_size,%off0,%circ0,%out) : (index,index,memref<256xi16>,memref<1024xf32>)->()
  call @concat_aux(%sample_size,%off1,%circ1,%out) : (index,index,memref<256xi16>,memref<1024xf32>)->()
  call @concat_aux(%sample_size,%off2,%circ2,%out) : (index,index,memref<256xi16>,memref<1024xf32>)->()
  call @concat_aux(%sample_size,%off3,%circ3,%out) : (index,index,memref<256xi16>,memref<1024xf32>)->()
  return
}
func @extract_samples(%in:memref<?xf32>,
                      %out:memref<?xi16>) {
  %0 = constant 0 : index
  %1 = constant 1 : index
  %sample_size = memref.dim %out,%0 : memref<?xi16>
  scf.for %idx=%0 to %sample_size step %1 {
    %x = memref.load %in[%idx] : memref<?xf32>
    %z = std.fptosi %x : f32 to i16
    memref.store %z,%out[%idx] : memref<?xi16>
  }
  return
}

func @zero_tensor_512_i16(%r:memref<512xi16>) {
  return
}

func @undef_tensor_1024_f32(%r:memref<1024xf32>) {
  return
}

func @undef_tensor_512_f32(%r:memref<512xf32>) {
  return
}

func @undef_tensor_1024_i32(%r:memref<1024xi32>) {
  return
}

func @undef_tensor_512_i16(%r:memref<512xi16>) {
  return
}

func @undef_tensor_256_i16(%r:memref<256xi16>) {
  return
}

func @undef_tensor_512_complexf32(%r:memref<512xcomplex<f32>>) {
  return
}

func @undef_tensor_1024_complexf32(%r:memref<1024xcomplex<f32>>) {
  return
}

func @undef_f32() -> (f32) {
  %c = constant 0.0: f32
  return %c: f32
}

func @undef_i1() -> (i1) {
  %c = constant 0: i1
  return %c: i1
}

func @select_tensor_1024_f32(%c:i1,%t:memref<1024xf32>,%f:memref<1024xf32>,%r:memref<1024xf32>) {
  %to_copy = select %c,%t,%f : memref<1024xf32>
  linalg.copy(%to_copy,%r) : memref<1024xf32>,memref<1024xf32>
  return
}

func @select_tensor_512_f32(%c:i1,%t:memref<512xf32>,%f:memref<512xf32>,%r:memref<512xf32>) {
  %to_copy = select %c,%t,%f : memref<512xf32>
  linalg.copy(%to_copy,%r) : memref<512xf32>,memref<512xf32>
  return
}

func @select_tensor_512_i16(%c:i1,%t:memref<512xi16>,%f:memref<512xi16>,%r:memref<512xi16>) {
  %to_copy = select %c,%t,%f : memref<512xi16>
  linalg.copy(%to_copy,%r) : memref<512xi16>,memref<512xi16>
  return
}

func @select_tensor_256_i16(%c:i1,%t:memref<256xi16>,%f:memref<256xi16>,%r:memref<256xi16>) {
  %to_copy = select %c,%t,%f : memref<256xi16>
  linalg.copy(%to_copy,%r) : memref<256xi16>,memref<256xi16>
  return
}

func @bzero_i16(%size:index,%m:memref<?xi16>) {
  %0 = constant 0 : index
  %1 = constant 1 : index
  scf.for %idx = %0 to %size step %1 {
    %zero = constant 0 : i16
    memref.store %zero, %m[%idx] : memref<?xi16>
  }
  return
}
func @bzero_f32(%size:index,%m:memref<?xf32>) {
  %0 = constant 0 : index
  %1 = constant 1 : index
  scf.for %idx = %0 to %size step %1 {
    %zero = constant 0.0 : f32
    memref.store %zero, %m[%idx] : memref<?xf32>
  }
  return
}
func @bzero_i16_256(%m:memref<256xi16>) {
  %256 = constant 256 : index
  %m1 = memref.cast %m : memref<256xi16> to memref<?xi16>
  call @bzero_i16(%256,%m1) : (index,memref<?xi16>)->()
  return
}
func @bzero_f32_512(%m:memref<512xf32>) {
  %512 = constant 512 : index
  %m1 = memref.cast %m : memref<512xf32> to memref<?xf32>
  call @bzero_f32(%512,%m1) : (index,memref<?xf32>)->()
  return
}
func @bzero_f32_1024(%m:memref<1024xf32>) {
  %1024 = constant 1024 : index
  %m1 = memref.cast %m : memref<1024xf32> to memref<?xf32>
  call @bzero_f32(%1024,%m1) : (index,memref<?xf32>)->()
  return
}
func @bzero_i16_1024(%m:memref<1024xi16>) {
  %1024 = constant 1024 : index
  %m1 = memref.cast %m : memref<1024xi16> to memref<?xi16>
  call @bzero_i16(%1024,%m1) : (index,memref<?xi16>)->()
  return
}
