// ../../mlir-lus/mlirlus just_lstm.mlir --normalize --convert-lus-to-sync --convert-sync-to-std | /home/hpompougnac/.cache/bazel/_bazel_hpompougnac/52c22a949d9769dd1d18ce3d575a5e05/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/mlir/tf-opt --canonicalize --sccp "-xla-legalize-tf=allow-partial-conversion use-tf2xla-fallback=true device-type=" --canonicalize | /home/hpompougnac/.cache/bazel/_bazel_hpompougnac/c21dee9b578a4b1c3f60a275119db250/execroot/iree_core/bazel-out/k8-opt/bin/iree/tools/iree-opt -iree-flow-hlo-to-hlo-preprocessing -iree-flow-extract-pad-from-conv --iree-codegen-hlo-to-linalg-on-tensors --linalg-fold-unit-extent-dims --canonicalize --iree-codegen-fusion-of-tensor-ops --cse --iree-codegen-hlo-to-linalg-on-tensors | ~/llvm-old/bin/mlir-opt --canonicalize --cse --func-bufferize --buffer-results-to-out-params --tensor-constant-bufferize --linalg-bufferize --linalg-detensorize | ../../mlir-prime/mlir-prime --bufferize-linalg-reshape | ~/llvm-old/bin/mlir-opt --tensor-bufferize --scf-bufferize --std-bufferize --buffer-deallocation --finalizing-bufferize --cse | ../../mlir-prime/mlir-prime --remove-copy-prime --prime-linalg-to-affine --canonicalize --convert-linalg-to-affine-loops --cse --loop-permutation-prime=permutation-map=2,3,4,5,0,1,6 --affine-loop-normalize --lower-affine | ~/llvm-old/bin/mlir-opt --convert-scf-to-std --test-math-polynomial-approximation --convert-linalg-to-llvm  --convert-complex-to-llvm --convert-std-to-llvm

lus.node @modulocounter(%mod: i32) -> (i1) {
  %c0 = constant 0: i32
  %c1 = constant 1: i32
  %bound = subi %mod, %c1: i32
  %time_cnt = lus.fby %c0 %time_cnt_mod: i32
  %time_cnt_incr = addi %time_cnt, %c1: i32
  %time_cnt_mod = remi_unsigned %time_cnt_incr, %mod: i32
  %lstm_clk = cmpi "eq", %time_cnt_mod, %bound: i32
  lus.yield (%lstm_clk: i1)
}

lus.node @lstm(%data: tensor<3x1xf32>) -> (tensor<3x100xf32>){
// stuff
  %o76 = constant dense<0.0>: tensor<100x400xf32>
  %o22 = constant dense<0.0>: tensor<1x400xf32>
  %o78 = constant dense<0.0>: tensor<400xf32>
  %split_dim = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %five = constant 5: i32
  %zero = constant dense<0.0>: tensor<3x100xf32>
  // Build a clock that is true every 5 cycles
  %lstm_clk = lus.instance inline @modulocounter(%five): (i32) -> (i1)
  // State and periodic reinit
  %tmp0 = lus.fby %zero %state0out: tensor<3x100xf32>
  %tmp1 = lus.fby %zero %lstm_out: tensor<3x100xf32>
  %v24 = select %lstm_clk, %zero, %tmp0 : tensor<3x100xf32>
  %v25 = select %lstm_clk, %zero, %tmp1 : tensor<3x100xf32>
  // LSTM core
  %v26 = "tf.MatMul"(%v24, %o76) {transpose_a = false, transpose_b = false} : (tensor<3x100xf32>, tensor<100x400xf32>) -> tensor<3x400xf32>
  %v28 = "tf.MatMul"(%data, %o22) {transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x400xf32>) -> tensor<3x400xf32>
  %v29 = "tf.AddV2"(%v28, %v26) : (tensor<3x400xf32>, tensor<3x400xf32>) -> tensor<3x400xf32>
  %v30 = "tf.BiasAdd"(%v29, %o78) {data_format = "NHWC"} : (tensor<3x400xf32>, tensor<400xf32>) -> tensor<3x400xf32>
  // split4
  %v31_0, %v31_1, %v31_2, %v31_3 = 
    "tf.Split"(%split_dim, %v30): (tensor<i32>, tensor<3x400xf32>) -> (tensor<3x100xf32>, tensor<3x100xf32>, tensor<3x100xf32>, tensor<3x100xf32>)
  //
  %v32 = "tf.Relu"(%v31_2): (tensor<3x100xf32>) -> tensor<3x100xf32>
  %v33 = "tf.Sigmoid"(%v31_0): (tensor<3x100xf32>) -> tensor<3x100xf32>
  %v34 = "tf.Mul"(%v33, %v32): (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
  %v35 = "tf.Sigmoid"(%v31_1): (tensor<3x100xf32>) -> tensor<3x100xf32>
  %v36 = "tf.Mul"(%v35, %v25): (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
  %lstm_out = "tf.AddV2"(%v36, %v34): (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
  %v40 = "tf.Relu"(%lstm_out): (tensor<3x100xf32>) -> tensor<3x100xf32>
  %v41 = "tf.Sigmoid"(%v31_3): (tensor<3x100xf32>) -> tensor<3x100xf32>
  %state0out = "tf.Mul"(%v41, %v40): (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
  // Output subsampling
  %subsampled = lus.when %lstm_clk %lstm_out: tensor<3x100xf32>
  lus.yield (%subsampled: tensor<3x100xf32>)
}


// lus.node @lstm(%data: tensor<3x1xf32>,%rst:i1) -> (tensor<3x100xf32>) {
//   %zero_tensor_cst = constant dense<0.0>: tensor<3x100xf32>
//   %tmp0 = lus.fby %zero_tensor_cst %state0out: tensor<3x100xf32>
//   %tmp1 = lus.fby %zero_tensor_cst %state1out: tensor<3x100xf32>
//   %v24 = select %rst, %zero_tensor_cst, %tmp0 : tensor<3x100xf32>
//   %v25 = select %rst, %zero_tensor_cst, %tmp1 : tensor<3x100xf32>
//   %o76 = constant dense<0.0>: tensor<100x400xf32>
//   %o22 = constant dense<0.0>: tensor<1x400xf32>
//   %o78 = constant dense<0.0>: tensor<400xf32>
//   %v26 = "tf.MatMul"(%v24, %o76) {device = "", transpose_a = false, transpose_b = false} : (tensor<3x100xf32>, tensor<100x400xf32>) -> tensor<3x400xf32>
//   %v28 =  "tf.MatMul"(%data, %o22) {device = "", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x400xf32>) -> tensor<3x400xf32>
//   %v29 = "tf.AddV2"(%v28, %v26) {device = ""} : (tensor<3x400xf32>, tensor<3x400xf32>) -> tensor<3x400xf32>
//   %v30 = "tf.BiasAdd"(%v29, %o78) {data_format = "NHWC", device = ""} : (tensor<3x400xf32>, tensor<400xf32>) -> tensor<3x400xf32>
//   //split4
//   %split_dim = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
//   %v31_0, %v31_1, %v31_2, %v31_3 = "tf.Split"(%split_dim, %v30): (tensor<i32>, tensor<3x400xf32>) -> (tensor<3x100xf32>, tensor<3x100xf32>, tensor<3x100xf32>, tensor<3x100xf32>)
//   //
//   %v32 = "tf.Relu"(%v31_2) {device = ""} : (tensor<3x100xf32>) -> tensor<3x100xf32>
//   %v33 = "tf.Sigmoid"(%v31_0): (tensor<3x100xf32>) -> tensor<3x100xf32>
//   %v34 = "tf.Mul"(%v33, %v32): (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
//   %v35 = "tf.Sigmoid"(%v31_1): (tensor<3x100xf32>) -> tensor<3x100xf32>
//   %v36 = "tf.Mul"(%v35, %v25): (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
//   %state1out = "tf.AddV2"(%v36, %v34) {device = ""} : (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
//   %v40 = "tf.Relu"(%state1out) {device = ""} : (tensor<3x100xf32>) -> tensor<3x100xf32>
//   %v41 = "tf.Sigmoid"(%v31_3): (tensor<3x100xf32>) -> tensor<3x100xf32>
//   %state0out = "tf.Mul"(%v41, %v40): (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>

//   lus.yield (%state1out: tensor<3x100xf32>)
// }
