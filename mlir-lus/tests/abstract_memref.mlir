// Gencode pour memref<?xi32> : shape contient -1
// MemRefCast

func private @tick() -> i32
func private @halt()

func private @aux(%arg0: i64, %arg1: (i64, i32, memref<?xi32>) -> (), %arg2: (i64, i32, memref<?xi32>) -> ()) {
  %c2_i32 = constant 2 : i32
  %c1_i32 = constant 1 : i32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c9223372036854775807 = constant 9223372036854775807 : index
  scf.for %arg3 = %c0 to %c9223372036854775807 step %c1 {
    %0 = alloc() : memref<5xi32>
    %b = memref_cast %0 : memref<5xi32> to memref<?xi32>
    call_indirect %arg1(%arg0, %c1_i32, %b) : (i64, i32, memref<?xi32>) -> ()
    %1 = tensor_load %0 : memref<5xi32>
    %2 = addi %1, %1 : tensor<5xi32>
    %c = memref_cast %0 : memref<5xi32> to memref<?xi32>
    call_indirect %arg2(%arg0, %c2_i32, %c) : (i64, i32, memref<?xi32>) -> ()
    %3 = call @tick() : () -> i32
  }
  call @halt() : () -> ()
  return
}
