func private @timeseries(i32, (i32, memref<3x1xf32>) -> (), (i32, memref<3x1xf32>) -> ())
func private @init_time() -> ()

func private @read_tensors(%pos: i32, %i: memref<3x1xf32>) { return }
func private @write_tensors(%pos: i32, %o: memref<3x1xf32>) { return }

func @main() {
  %inst = constant 1: i32
  %sndi = constant @read_tensors : (i32, memref<3x1xf32>)->()
  %sndo = constant @write_tensors : (i32, memref<3x1xf32>)->()
  call @init_time(): () -> ()
  call @timeseries(%inst, %sndi, %sndo): (i32, (i32, memref<3x1xf32>) -> (), (i32, memref<3x1xf32>) -> ()) -> ()
  return
}
