// These are functions that are better specified in memref domain,
// mostly because they call functions that have output arguments
// that are tensors of unknown size (which cannot automatically
// be lowered while ensuring there are no alloc operations that are
// not deallocated.
//
// To interface with the tensor-based functions, I have to help
// translation by performing half of it by hand. In doing this,
// don't forget the output memref at at the end of the argument list.
func private @complex_tensor_float_product(memref<?xcomplex<f32>>,f32,memref<?xcomplex<f32>>)
func private @bf1_up(%tw:complex<f32>,%i0:complex<f32>,%i1:complex<f32>)->complex<f32>
func private @bf1_down(%tw:complex<f32>,%i0:complex<f32>,%i1:complex<f32>)->complex<f32>
func private @bitrev(%perm:memref<?xi32>,%bf_res:memref<?xcomplex<f32>>,%res:memref<?xcomplex<f32>>)
func private @complex_fprod(%i:complex<f32>,%f:f32)->(complex<f32>)
func private @complex_swap(%i:complex<f32>)->(complex<f32>)

func @compute_twiddles(%res:memref<?xcomplex<f32>>){
  // Integer constants
  %0 = constant 0 : index 
  %1 = constant 1 : index 
  %2 = constant 2 : index
  // FP constants
  %f0 = constant 0. : f32
  %pi = constant 3.14159265358979 : f32
  // The memref where the twiddles are built
  %size = memref.dim %res,%0 : memref<?xcomplex<f32>>
  %sz2 = divi_signed %size,%2 : index
  scf.while (%span = %1) : (index) -> (index) {
    // while %span <= (%size/2)
    %cond = cmpi "sle",%span,%sz2 : index
    scf.condition(%cond) %span : index
  } do {
  ^bb0(%span: index):
    %spani = std.index_cast %span:index to i32
    %spanf = std.sitofp %spani:i32 to f32
    %x1 = divf %pi,%spanf : f32
    %primitive_root = subf %f0, %x1 : f32 
    scf.for %i = %0 to %span step %1 {
      // Compute the complex twiddle value
      %ii = std.index_cast %i:index to i32
      %if = std.sitofp %ii:i32 to f32
      %x2 = mulf %if, %primitive_root : f32
      %re = math.cos %x2 : f32
      %im = math.sin %x2 : f32
      %twiddle = complex.create %re, %im : complex<f32>
      // Compute the index %span+%i
      %idx = addi %span, %i : index
      memref.store %twiddle, %res[%idx] : memref<?xcomplex<f32>>
    }
    // %span := %span*2
    %span_next = muli %span, %2 : index
    scf.yield %span_next:index
  }
  return
}


func @fft_normalize(%data:memref<?xcomplex<f32>>,%res:memref<?xcomplex<f32>>) {
  %0 = constant 0 : index
  %size = memref.dim %data, %0 : memref<?xcomplex<f32>>
  // Compute the multiplication factor
  %tmp1 = std.index_cast %size:index to i32
  %tmp2= std.sitofp %tmp1:i32 to f32
  %f1 = constant 1.0 : f32
  %norm_val = std.divf %f1, %tmp2 : f32
  call @complex_tensor_float_product(%data,%norm_val,%res):(memref<?xcomplex<f32>>,f32,memref<?xcomplex<f32>>)->()
  return
}

#map = affine_map<(d0)[s0, s1]->(d0 * s1 + s0)>
func @fft_aux(%twid:memref<?xcomplex<f32>>,%data:memref<?xcomplex<f32>>,%res:memref<?xcomplex<f32>>) {
  %0 = constant 0 : index
  %1 = constant 1 : index
  %2 = constant 2 : index
  %size = memref.dim %data, %0 : memref<?xcomplex<f32>>

  // Local data storage, filled in with %data to avoid deletion
  %tmp = memref.alloc(%size) : memref<?xcomplex<f32>>
  linalg.copy(%data,%tmp) : memref<?xcomplex<f32>>,memref<?xcomplex<f32>>

  // Butterfly steps
  %a = scf.while(%curr_size = %size) : (index)->(index) {
    // Check the size is a power of 2
//    %tmpidx = std.and %curr_size, %1 : index
//    %evensize = cmpi "eq",%tmpidx,%0 : index
//    assert %evensize, "@fft: size of input data is not power of 2."
    // While I still have to perform one step
    %cond = cmpi "sge",%curr_size,%2 : index
    scf.condition(%cond) %curr_size : index
  } do {
  ^bb0(%curr_size: index):
    // Half the current size
    %halfsize = divi_signed %curr_size,%2 : index
    // Extract the half-data memref of the twiddles
    %twid1 = memref.subview %twid[%halfsize][%halfsize][%1] : memref<?xcomplex<f32>> to memref<?xcomplex<f32>,#map>
    // Number of iterations to make here
    %iter_no = divi_signed %size, %curr_size : index
    scf.for %iter = %0 to %iter_no step %1 {
      // The input data is in %tmp, I will write it to %res
      // Extract the data half-vectors
      %base0 = muli %iter, %curr_size : index
      %base1 = addi %base0, %halfsize : index
      %tmp0 = memref.subview %tmp[%base0][%halfsize][%1] : memref<?xcomplex<f32>> to memref<?xcomplex<f32>,#map>
      %tmp1 = memref.subview %tmp[%base1][%halfsize][%1] : memref<?xcomplex<f32>> to memref<?xcomplex<f32>,#map>
      %res0 = memref.subview %res[%base0][%halfsize][%1] : memref<?xcomplex<f32>> to memref<?xcomplex<f32>,#map>
      %res1 = memref.subview %res[%base1][%halfsize][%1] : memref<?xcomplex<f32>> to memref<?xcomplex<f32>,#map>

      // Compute one butterfly of size %curr_size. The output is stored
      // in the tmp memref.
      // Here, std.yield accepts only one argument, so I can't use
      // a single dynamic_tensor_from_elements...
      scf.for %i = %0 to %halfsize step %1 {
        %tw = memref.load %twid1[%i] : memref<?xcomplex<f32>,#map>
	%i0 = memref.load %tmp0[%i] : memref<?xcomplex<f32>,#map>
	%i1 = memref.load %tmp1[%i] : memref<?xcomplex<f32>,#map>
        %o0 = call @bf1_up(%tw,%i0,%i1):(complex<f32>,complex<f32>,complex<f32>)->complex<f32>
        %o1 = call @bf1_down(%tw,%i0,%i1):(complex<f32>,complex<f32>,complex<f32>)->complex<f32>
	memref.store %o0,%res0[%i] : memref<?xcomplex<f32>,#map>
	memref.store %o1,%res1[%i] : memref<?xcomplex<f32>,#map>
      }
    }
    // Copy back the data from %res to %tmp
    linalg.copy(%res,%tmp) : memref<?xcomplex<f32>>,memref<?xcomplex<f32>>
    scf.yield %halfsize:index
  }

  memref.dealloc %tmp : memref<?xcomplex<f32>>
  return
}

func @fft(%perm:memref<?xi32>,%twid:memref<?xcomplex<f32>>,%data:memref<?xcomplex<f32>>,%res:memref<?xcomplex<f32>>) {
  %0 = constant 0 : index
  %size = memref.dim %data,%0 : memref<?xcomplex<f32>>
  %bf_result = memref.alloc(%size) : memref<?xcomplex<f32>>
  %brp_result = memref.alloc(%size) : memref<?xcomplex<f32>>


  call @fft_aux(%twid,%data,%bf_result) : (memref<?xcomplex<f32>>,memref<?xcomplex<f32>>,memref<?xcomplex<f32>>)->()
  call @bitrev(%perm,%bf_result,%brp_result) : (memref<?xi32>,memref<?xcomplex<f32>>,memref<?xcomplex<f32>>)->()
  call @fft_normalize(%brp_result,%res) : (memref<?xcomplex<f32>>,memref<?xcomplex<f32>>)->()
  memref.dealloc %bf_result : memref<?xcomplex<f32>>
  memref.dealloc %brp_result : memref<?xcomplex<f32>>
  return
} 

func @ifft(%perm:memref<?xi32>,%twid:memref<?xcomplex<f32>>,%data:memref<?xcomplex<f32>>,%res:memref<?xcomplex<f32>>) {
  %0 = constant 0 : index
  %1 = constant 1 : index
  %size = memref.dim %data, %0 : memref<?xcomplex<f32>>
  %swp_data = memref.alloc(%size) : memref<?xcomplex<f32>>
  %fft_res = memref.alloc(%size) : memref<?xcomplex<f32>>
  %swp2_data = memref.alloc(%size) : memref<?xcomplex<f32>>
  // Swap re<->im in all data
  scf.for %idx = %0 to %size step %1 {
    %x = memref.load %data[%idx] : memref<?xcomplex<f32>>
    %y = call  @complex_swap(%x) : (complex<f32>)->complex<f32>
    memref.store %y, %swp_data[%idx] : memref<?xcomplex<f32>>
  }
  // Call FFT
  call @fft(%perm,%twid,%swp_data,%fft_res): (memref<?xi32>,memref<?xcomplex<f32>>,memref<?xcomplex<f32>>,memref<?xcomplex<f32>>)->()
  // Swap back im<->re in all data
  scf.for %idx = %0 to %size step %1 {
    %x = memref.load %fft_res[%idx] : memref<?xcomplex<f32>>
    %y = call @complex_swap(%x) : (complex<f32>)->complex<f32>
    memref.store %y, %swp2_data[%idx] : memref<?xcomplex<f32>>
  }  
  // Normalize - potential for loop fusion with the previous loop
  // to avoid buffer synthesis
  %tmp1 = std.index_cast %size:index to i32
  %norm_val= std.sitofp %tmp1:i32 to f32
  scf.for %idx = %0 to %size step %1 {
    %x = memref.load %swp2_data[%idx] : memref<?xcomplex<f32>>
    %y = call @complex_fprod(%x,%norm_val) : (complex<f32>,f32)->(complex<f32>)
    memref.store %y, %res[%idx] : memref<?xcomplex<f32>>
  }
  return
}
