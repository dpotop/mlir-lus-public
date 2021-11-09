func private @bzero_i16_256()->tensor<256xi16>

lus.node @pitch(%sndbufin:tensor<256xi16>)->(tensor<256xi16>) {
  %c0 = call @bzero_i16_256() : ()->tensor<256xi16>
  %circ2 = lus.fby %c0 %sndbufin : tensor<256xi16>
  lus.yield(%circ2:tensor<256xi16> )
}

//----------------------------------------------------------------
// On peut faire ceci même si v et x ont une horloge
// différente de l'horloge de base.
//----------------------------------------------------------------

// x = read_input(0 when b)
// v = init fby x ;
// f(v) ;

// %init = ...
// %zero = ...
// scf.for %cnt %0 to %beaucoup step %1 iter_args(%v = %init)->(tensor<256xi16>tensor<256xi16>)  {
//   %b = call @read_bool() : () -> (i1)
//   %v_next = scf.if %b {
//     %x = call @read_input(%zero) : (i32) -> (tensor<256xi16>)
//     call @f(%v) : (tensor<256xi16>)->()
//     scf.yield %x : tensor<256xi16>
//   } else {
//     // Maintain the old value
//     scf.yield %v : tensor<256xi16>
//   }
//   scf.yield %v_next: tensor<256xi16>
// }
