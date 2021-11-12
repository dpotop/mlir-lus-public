// This is the entry point of the application, whose sole
// objective is to initialize HW and to set the I/O
// functions to the algorithm specified in the lus dialect.

func private @open_kbd()->()
func private @read_kbd(i32, memref<1xi8>) attributes {llvm.emit_c_interface}
func private @read_samples(i32, memref<512xi16>) attributes {llvm.emit_c_interface}
func private @write_samples(i32, memref<512xi16>) attributes {llvm.emit_c_interface}
func private @pitch(i32, (i32, memref<1xi8>) -> (), (i32, memref<512xi16>) -> (), (i32, memref<512xi16>) -> ())

func @main()->() {
  // Init keyboard HW
  call @open_kbd() : ()->()
  %inst = constant 1: i32
  // This function never terminates
  %kbd = constant @read_kbd : (i32, memref<1xi8>)->()
  %sndi = constant @read_samples : (i32, memref<512xi16>)->()
  %sndo = constant @write_samples : (i32, memref<512xi16>)->()
  call @pitch(%inst,%kbd,%sndi,%sndo):(i32,(i32, memref<1xi8>)->(),(i32, memref<512xi16>)->(),(i32, memref<512xi16>)->())->()
  return
}
