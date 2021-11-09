//===============================================================
// scf.if seems to be well-handled by bufferization:
// ~/llvm/bin/mlir-opt --scf-bufferize --func-bufferize \
//       --std-bufferize --finalizing-bufferize try2.mlir
// The resulting code is even quite lean - there is a step that
// removes unneeded copies inside --finalizing-bufferize
func @test(%x:tensor<1024xcomplex<f32>>)->(tensor<1024xcomplex<f32>>){
  // Integer constants
  %0 = constant 0 : index 
  %1 = constant 1 : index
  %true = constant true
  %y = scf.if %true -> tensor<1024xcomplex<f32>> {
    scf.yield %x:tensor<1024xcomplex<f32>>
  } else {
    scf.yield %x:tensor<1024xcomplex<f32>>
  } 
  return %y: tensor<1024xcomplex<f32>>
}

//===============================================================
// scf.while (more precisely, scf.condition) still does not
// pass mlir-opt --scf-bufferize
// 
//   scf.while (%cycle = %0) : (index) -> (index) {
//     // while true
//     %true = constant true
//     scf.condition(%true) %cycle : index
//   } do {
//   ^bb0(%cycle: index):
//     // %cycle := %cycle + 1
//     %cycle_next = addi %cycle, %1 : index
//     scf.yield %cycle_next:index
//   }
//   return
// }
