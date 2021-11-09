func private @read_tensors(i32, memref<1x224x224x3xf32>) attributes {llvm.emit_c_interface}
func private @write_tensors(i32, memref<1x1000xf32>) attributes {llvm.emit_c_interface}
func private @resnet(i32, (i32, memref<1x224x224x3xf32>) -> (), (i32, memref<1x1000xf32>) -> ())
func private @init_time() -> ()

func @main() {
  %inst = constant 1: i32
  %sndi = constant @read_tensors : (i32, memref<1x224x224x3xf32>)->()
  %sndo = constant @write_tensors : (i32, memref<1x1000xf32>)->()
  call @init_time(): () -> ()
  call @resnet(%inst, %sndi, %sndo): (i32, (i32, memref<1x224x224x3xf32>) -> (), (i32, memref<1x1000xf32>) -> ()) -> ()
  return
}
