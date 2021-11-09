//===================================================================
//     ~/llvm/bin/mlir-opt --scf-bufferize --func-bufferize \
//              --std-bufferize --finalizing-bufferize try.mlir |\
//           ~/llvm/bin/mlir-opt --buffer-results-to-out-params |\
//           ~/llvm/bin/mlir-opt --buffer-deallocation

func private @bzero()->tensor<256xi16>
func private @read()->tensor<256xi16>
func private @write(tensor<256xi16>)->()

func @pitch()->() {
  %0    = constant 0 : index
  %1    = constant 1 : index
  %1000 = constant 1000 : index
  %init   = call @bzero() : ()->tensor<256xi16>
  scf.for %cnt = %0 to %1000 step %1
    iter_args(%fby0 = %init,%fby1 = %init)->(tensor<256xi16>,tensor<256xi16>) {
    %x = call @read() : ()->tensor<256xi16>
    call @write(%fby0) : (tensor<256xi16>)->()
    scf.yield %fby1,%x: tensor<256xi16>,tensor<256xi16>
  }
  return
}
