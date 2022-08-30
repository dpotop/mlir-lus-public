
lus.node @resnet(%v0: tensor<1x224x224x3xf32>) -> (tensor<1x1000xf32>) {

%v1 = "tf.Const"() {value = dense<[[0, 0], [3, 3], [3, 3], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
%v2 = "tf.Pad"(%v0, %v1) {data_format = "NHWC"} : (tensor<1x224x224x3xf32>, tensor<4x2xi32>) -> tensor<1x230x230x3xf32>
%v6 = "tf.Const"() {value = dense<0.0> : tensor<7x7x3x64xf32>} : () -> tensor<7x7x3x64xf32>
%v7 = "tf.Conv2D"(%v2, %v6) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x230x230x3xf32>, tensor<7x7x3x64xf32>) -> tensor<1x112x112x64xf32>
%v4 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v5 = "tf.BiasAdd"(%v7, %v4) {data_format = "NHWC"} : (tensor<1x112x112x64xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
%v3 = "tf.Identity"(%v5) {} : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
%v8 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v9 = "tf.Const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v10 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v11 = "tf.Const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v12, %v13, %v14, %v15, %v16, %v17 = "tf.FusedBatchNormV3"(%v3, %v8, %v9, %v10, %v11) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x112x112x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x112x112x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<*xf32>)
%v18 = "tf.Relu"(%v12) {} : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
%v19 = "tf.Const"() {value = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
%v20 = "tf.Pad"(%v18, %v19) {data_format = "NHWC"} : (tensor<1x112x112x64xf32>, tensor<4x2xi32>) -> tensor<1x114x114x64xf32>
%v21 = "tf.MaxPool"(%v20) {ksize = [1, 3, 3, 1], padding = "VALID", strides = [1, 2, 2, 1], data_format = "NHWC", explicit_paddings = []} : (tensor<1x114x114x64xf32>) -> tensor<1x56x56x64xf32>
%v25 = "tf.Const"() {value = dense<0.0> : tensor<1x1x64x256xf32>} : () -> tensor<1x1x64x256xf32>
%v26 = "tf.Conv2D"(%v21, %v25) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32>
%v23 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v24 = "tf.BiasAdd"(%v26, %v23) {data_format = "NHWC"} : (tensor<1x56x56x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
%v22 = "tf.Identity"(%v24) {} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
%v27 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v28 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v29 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v30 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v31, %v32, %v33, %v34, %v35, %v36 = "tf.FusedBatchNormV3"(%v22, %v27, %v28, %v29, %v30) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v40 = "tf.Const"() {value = dense<0.0> : tensor<1x1x64x64xf32>} : () -> tensor<1x1x64x64xf32>
%v41 = "tf.Conv2D"(%v21, %v40) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) -> tensor<1x56x56x64xf32>
%v38 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v39 = "tf.BiasAdd"(%v41, %v38) {data_format = "NHWC"} : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
%v37 = "tf.Identity"(%v39) {} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
%v42 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v43 = "tf.Const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v44 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v45 = "tf.Const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v46, %v47, %v48, %v49, %v50, %v51 = "tf.FusedBatchNormV3"(%v37, %v42, %v43, %v44, %v45) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<*xf32>)
%v52 = "tf.Relu"(%v46) {} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
%v56 = "tf.Const"() {value = dense<0.0> : tensor<3x3x64x64xf32>} : () -> tensor<3x3x64x64xf32>
%v57 = "tf.Conv2D"(%v52, %v56) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
%v54 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v55 = "tf.BiasAdd"(%v57, %v54) {data_format = "NHWC"} : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
%v53 = "tf.Identity"(%v55) {} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
%v58 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v59 = "tf.Const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v60 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v61 = "tf.Const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v62, %v63, %v64, %v65, %v66, %v67 = "tf.FusedBatchNormV3"(%v53, %v58, %v59, %v60, %v61) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<*xf32>)
%v68 = "tf.Relu"(%v62) {} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
%v72 = "tf.Const"() {value = dense<0.0> : tensor<1x1x64x256xf32>} : () -> tensor<1x1x64x256xf32>
%v73 = "tf.Conv2D"(%v68, %v72) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32>
%v70 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v71 = "tf.BiasAdd"(%v73, %v70) {data_format = "NHWC"} : (tensor<1x56x56x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
%v69 = "tf.Identity"(%v71) {} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
%v74 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v75 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v76 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v77 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v78, %v79, %v80, %v81, %v82, %v83 = "tf.FusedBatchNormV3"(%v69, %v74, %v75, %v76, %v77) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v84 = "tf.AddV2"(%v31, %v78) {} : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
%v85 = "tf.Relu"(%v84) {} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
%v89 = "tf.Const"() {value = dense<0.0> : tensor<1x1x256x64xf32>} : () -> tensor<1x1x256x64xf32>
%v90 = "tf.Conv2D"(%v85, %v89) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32>) -> tensor<1x56x56x64xf32>
%v87 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v88 = "tf.BiasAdd"(%v90, %v87) {data_format = "NHWC"} : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
%v86 = "tf.Identity"(%v88) {} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
%v91 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v92 = "tf.Const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v93 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v94 = "tf.Const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v95, %v96, %v97, %v98, %v99, %v100 = "tf.FusedBatchNormV3"(%v86, %v91, %v92, %v93, %v94) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<*xf32>)
%v101 = "tf.Relu"(%v95) {} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
%v105 = "tf.Const"() {value = dense<0.0> : tensor<3x3x64x64xf32>} : () -> tensor<3x3x64x64xf32>
%v106 = "tf.Conv2D"(%v101, %v105) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
%v103 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v104 = "tf.BiasAdd"(%v106, %v103) {data_format = "NHWC"} : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
%v102 = "tf.Identity"(%v104) {} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
%v107 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v108 = "tf.Const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v109 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v110 = "tf.Const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v111, %v112, %v113, %v114, %v115, %v116 = "tf.FusedBatchNormV3"(%v102, %v107, %v108, %v109, %v110) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<*xf32>)
%v117 = "tf.Relu"(%v111) {} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
%v121 = "tf.Const"() {value = dense<0.0> : tensor<1x1x64x256xf32>} : () -> tensor<1x1x64x256xf32>
%v122 = "tf.Conv2D"(%v117, %v121) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32>
%v119 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v120 = "tf.BiasAdd"(%v122, %v119) {data_format = "NHWC"} : (tensor<1x56x56x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
%v118 = "tf.Identity"(%v120) {} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
%v123 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v124 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v125 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v126 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v127, %v128, %v129, %v130, %v131, %v132 = "tf.FusedBatchNormV3"(%v118, %v123, %v124, %v125, %v126) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v133 = "tf.AddV2"(%v85, %v127) {} : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
%v134 = "tf.Relu"(%v133) {} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
%v138 = "tf.Const"() {value = dense<0.0> : tensor<1x1x256x64xf32>} : () -> tensor<1x1x256x64xf32>
%v139 = "tf.Conv2D"(%v134, %v138) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32>) -> tensor<1x56x56x64xf32>
%v136 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v137 = "tf.BiasAdd"(%v139, %v136) {data_format = "NHWC"} : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
%v135 = "tf.Identity"(%v137) {} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
%v140 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v141 = "tf.Const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v142 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v143 = "tf.Const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v144, %v145, %v146, %v147, %v148, %v149 = "tf.FusedBatchNormV3"(%v135, %v140, %v141, %v142, %v143) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<*xf32>)
%v150 = "tf.Relu"(%v144) {} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
%v154 = "tf.Const"() {value = dense<0.0> : tensor<3x3x64x64xf32>} : () -> tensor<3x3x64x64xf32>
%v155 = "tf.Conv2D"(%v150, %v154) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
%v152 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v153 = "tf.BiasAdd"(%v155, %v152) {data_format = "NHWC"} : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
%v151 = "tf.Identity"(%v153) {} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
%v156 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v157 = "tf.Const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v158 = "tf.Const"() {value = dense<0.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v159 = "tf.Const"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
%v160, %v161, %v162, %v163, %v164, %v165 = "tf.FusedBatchNormV3"(%v151, %v156, %v157, %v158, %v159) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<*xf32>)
%v166 = "tf.Relu"(%v160) {} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
%v170 = "tf.Const"() {value = dense<0.0> : tensor<1x1x64x256xf32>} : () -> tensor<1x1x64x256xf32>
%v171 = "tf.Conv2D"(%v166, %v170) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32>
%v168 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v169 = "tf.BiasAdd"(%v171, %v168) {data_format = "NHWC"} : (tensor<1x56x56x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
%v167 = "tf.Identity"(%v169) {} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
%v172 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v173 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v174 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v175 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v176, %v177, %v178, %v179, %v180, %v181 = "tf.FusedBatchNormV3"(%v167, %v172, %v173, %v174, %v175) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v182 = "tf.AddV2"(%v134, %v176) {} : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
%v183 = "tf.Relu"(%v182) {} : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
%v187 = "tf.Const"() {value = dense<0.0> : tensor<1x1x256x512xf32>} : () -> tensor<1x1x256x512xf32>
%v188 = "tf.Conv2D"(%v183, %v187) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x56x56x256xf32>, tensor<1x1x256x512xf32>) -> tensor<1x28x28x512xf32>
%v185 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v186 = "tf.BiasAdd"(%v188, %v185) {data_format = "NHWC"} : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
%v184 = "tf.Identity"(%v186) {} : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
%v189 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v190 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v191 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v192 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v193, %v194, %v195, %v196, %v197, %v198 = "tf.FusedBatchNormV3"(%v184, %v189, %v190, %v191, %v192) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<*xf32>)
%v202 = "tf.Const"() {value = dense<0.0> : tensor<1x1x256x128xf32>} : () -> tensor<1x1x256x128xf32>
%v203 = "tf.Conv2D"(%v183, %v202) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x56x56x256xf32>, tensor<1x1x256x128xf32>) -> tensor<1x28x28x128xf32>
%v200 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v201 = "tf.BiasAdd"(%v203, %v200) {data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
%v199 = "tf.Identity"(%v201) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v204 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v205 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v206 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v207 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v208, %v209, %v210, %v211, %v212, %v213 = "tf.FusedBatchNormV3"(%v199, %v204, %v205, %v206, %v207) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<*xf32>)
%v214 = "tf.Relu"(%v208) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v218 = "tf.Const"() {value = dense<0.0> : tensor<3x3x128x128xf32>} : () -> tensor<3x3x128x128xf32>
%v219 = "tf.Conv2D"(%v214, %v218) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
%v216 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v217 = "tf.BiasAdd"(%v219, %v216) {data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
%v215 = "tf.Identity"(%v217) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v220 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v221 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v222 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v223 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v224, %v225, %v226, %v227, %v228, %v229 = "tf.FusedBatchNormV3"(%v215, %v220, %v221, %v222, %v223) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<*xf32>)
%v230 = "tf.Relu"(%v224) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v234 = "tf.Const"() {value = dense<0.0> : tensor<1x1x128x512xf32>} : () -> tensor<1x1x128x512xf32>
%v235 = "tf.Conv2D"(%v230, %v234) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) -> tensor<1x28x28x512xf32>
%v232 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v233 = "tf.BiasAdd"(%v235, %v232) {data_format = "NHWC"} : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
%v231 = "tf.Identity"(%v233) {} : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
%v236 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v237 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v238 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v239 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v240, %v241, %v242, %v243, %v244, %v245 = "tf.FusedBatchNormV3"(%v231, %v236, %v237, %v238, %v239) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<*xf32>)
%v246 = "tf.AddV2"(%v193, %v240) {} : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
%v247 = "tf.Relu"(%v246) {} : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
%v251 = "tf.Const"() {value = dense<0.0> : tensor<1x1x512x128xf32>} : () -> tensor<1x1x512x128xf32>
%v252 = "tf.Conv2D"(%v247, %v251) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) -> tensor<1x28x28x128xf32>
%v249 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v250 = "tf.BiasAdd"(%v252, %v249) {data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
%v248 = "tf.Identity"(%v250) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v253 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v254 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v255 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v256 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v257, %v258, %v259, %v260, %v261, %v262 = "tf.FusedBatchNormV3"(%v248, %v253, %v254, %v255, %v256) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<*xf32>)
%v263 = "tf.Relu"(%v257) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v267 = "tf.Const"() {value = dense<0.0> : tensor<3x3x128x128xf32>} : () -> tensor<3x3x128x128xf32>
%v268 = "tf.Conv2D"(%v263, %v267) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
%v265 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v266 = "tf.BiasAdd"(%v268, %v265) {data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
%v264 = "tf.Identity"(%v266) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v269 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v270 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v271 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v272 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v273, %v274, %v275, %v276, %v277, %v278 = "tf.FusedBatchNormV3"(%v264, %v269, %v270, %v271, %v272) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<*xf32>)
%v279 = "tf.Relu"(%v273) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v283 = "tf.Const"() {value = dense<0.0> : tensor<1x1x128x512xf32>} : () -> tensor<1x1x128x512xf32>
%v284 = "tf.Conv2D"(%v279, %v283) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) -> tensor<1x28x28x512xf32>
%v281 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v282 = "tf.BiasAdd"(%v284, %v281) {data_format = "NHWC"} : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
%v280 = "tf.Identity"(%v282) {} : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
%v285 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v286 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v287 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v288 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v289, %v290, %v291, %v292, %v293, %v294 = "tf.FusedBatchNormV3"(%v280, %v285, %v286, %v287, %v288) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<*xf32>)
%v295 = "tf.AddV2"(%v247, %v289) {} : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
%v296 = "tf.Relu"(%v295) {} : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
%v300 = "tf.Const"() {value = dense<0.0> : tensor<1x1x512x128xf32>} : () -> tensor<1x1x512x128xf32>
%v301 = "tf.Conv2D"(%v296, %v300) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) -> tensor<1x28x28x128xf32>
%v298 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v299 = "tf.BiasAdd"(%v301, %v298) {data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
%v297 = "tf.Identity"(%v299) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v302 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v303 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v304 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v305 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v306, %v307, %v308, %v309, %v310, %v311 = "tf.FusedBatchNormV3"(%v297, %v302, %v303, %v304, %v305) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<*xf32>)
%v312 = "tf.Relu"(%v306) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v316 = "tf.Const"() {value = dense<0.0> : tensor<3x3x128x128xf32>} : () -> tensor<3x3x128x128xf32>
%v317 = "tf.Conv2D"(%v312, %v316) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
%v314 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v315 = "tf.BiasAdd"(%v317, %v314) {data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
%v313 = "tf.Identity"(%v315) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v318 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v319 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v320 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v321 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v322, %v323, %v324, %v325, %v326, %v327 = "tf.FusedBatchNormV3"(%v313, %v318, %v319, %v320, %v321) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<*xf32>)
%v328 = "tf.Relu"(%v322) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v332 = "tf.Const"() {value = dense<0.0> : tensor<1x1x128x512xf32>} : () -> tensor<1x1x128x512xf32>
%v333 = "tf.Conv2D"(%v328, %v332) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) -> tensor<1x28x28x512xf32>
%v330 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v331 = "tf.BiasAdd"(%v333, %v330) {data_format = "NHWC"} : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
%v329 = "tf.Identity"(%v331) {} : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
%v334 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v335 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v336 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v337 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v338, %v339, %v340, %v341, %v342, %v343 = "tf.FusedBatchNormV3"(%v329, %v334, %v335, %v336, %v337) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<*xf32>)
%v344 = "tf.AddV2"(%v296, %v338) {} : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
%v345 = "tf.Relu"(%v344) {} : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
%v349 = "tf.Const"() {value = dense<0.0> : tensor<1x1x512x128xf32>} : () -> tensor<1x1x512x128xf32>
%v350 = "tf.Conv2D"(%v345, %v349) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) -> tensor<1x28x28x128xf32>
%v347 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v348 = "tf.BiasAdd"(%v350, %v347) {data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
%v346 = "tf.Identity"(%v348) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v351 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v352 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v353 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v354 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v355, %v356, %v357, %v358, %v359, %v360 = "tf.FusedBatchNormV3"(%v346, %v351, %v352, %v353, %v354) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<*xf32>)
%v361 = "tf.Relu"(%v355) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v365 = "tf.Const"() {value = dense<0.0> : tensor<3x3x128x128xf32>} : () -> tensor<3x3x128x128xf32>
%v366 = "tf.Conv2D"(%v361, %v365) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
%v363 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v364 = "tf.BiasAdd"(%v366, %v363) {data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
%v362 = "tf.Identity"(%v364) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v367 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v368 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v369 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v370 = "tf.Const"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
%v371, %v372, %v373, %v374, %v375, %v376 = "tf.FusedBatchNormV3"(%v362, %v367, %v368, %v369, %v370) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<*xf32>)
%v377 = "tf.Relu"(%v371) {} : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
%v381 = "tf.Const"() {value = dense<0.0> : tensor<1x1x128x512xf32>} : () -> tensor<1x1x128x512xf32>
%v382 = "tf.Conv2D"(%v377, %v381) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) -> tensor<1x28x28x512xf32>
%v379 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v380 = "tf.BiasAdd"(%v382, %v379) {data_format = "NHWC"} : (tensor<1x28x28x512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
%v378 = "tf.Identity"(%v380) {} : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
%v383 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v384 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v385 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v386 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v387, %v388, %v389, %v390, %v391, %v392 = "tf.FusedBatchNormV3"(%v378, %v383, %v384, %v385, %v386) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<*xf32>)
%v393 = "tf.AddV2"(%v345, %v387) {} : (tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
%v394 = "tf.Relu"(%v393) {} : (tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
%v398 = "tf.Const"() {value = dense<0.0> : tensor<1x1x512x1024xf32>} : () -> tensor<1x1x512x1024xf32>
%v399 = "tf.Conv2D"(%v394, %v398) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x28x28x512xf32>, tensor<1x1x512x1024xf32>) -> tensor<1x14x14x1024xf32>
%v396 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v397 = "tf.BiasAdd"(%v399, %v396) {data_format = "NHWC"} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
%v395 = "tf.Identity"(%v397) {} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v400 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v401 = "tf.Const"() {value = dense<1.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v402 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v403 = "tf.Const"() {value = dense<1.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v404, %v405, %v406, %v407, %v408, %v409 = "tf.FusedBatchNormV3"(%v395, %v400, %v401, %v402, %v403) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<*xf32>)
%v413 = "tf.Const"() {value = dense<0.0> : tensor<1x1x512x256xf32>} : () -> tensor<1x1x512x256xf32>
%v414 = "tf.Conv2D"(%v394, %v413) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x28x28x512xf32>, tensor<1x1x512x256xf32>) -> tensor<1x14x14x256xf32>
%v411 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v412 = "tf.BiasAdd"(%v414, %v411) {data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
%v410 = "tf.Identity"(%v412) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v415 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v416 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v417 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v418 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v419, %v420, %v421, %v422, %v423, %v424 = "tf.FusedBatchNormV3"(%v410, %v415, %v416, %v417, %v418) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v425 = "tf.Relu"(%v419) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v429 = "tf.Const"() {value = dense<0.0> : tensor<3x3x256x256xf32>} : () -> tensor<3x3x256x256xf32>
%v430 = "tf.Conv2D"(%v425, %v429) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
%v427 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v428 = "tf.BiasAdd"(%v430, %v427) {data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
%v426 = "tf.Identity"(%v428) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v431 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v432 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v433 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v434 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v435, %v436, %v437, %v438, %v439, %v440 = "tf.FusedBatchNormV3"(%v426, %v431, %v432, %v433, %v434) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v441 = "tf.Relu"(%v435) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v445 = "tf.Const"() {value = dense<0.0> : tensor<1x1x256x1024xf32>} : () -> tensor<1x1x256x1024xf32>
%v446 = "tf.Conv2D"(%v441, %v445) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
%v443 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v444 = "tf.BiasAdd"(%v446, %v443) {data_format = "NHWC"} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
%v442 = "tf.Identity"(%v444) {} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v447 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v448 = "tf.Const"() {value = dense<1.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v449 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v450 = "tf.Const"() {value = dense<1.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v451, %v452, %v453, %v454, %v455, %v456 = "tf.FusedBatchNormV3"(%v442, %v447, %v448, %v449, %v450) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<*xf32>)
%v457 = "tf.AddV2"(%v404, %v451) {} : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v458 = "tf.Relu"(%v457) {} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v462 = "tf.Const"() {value = dense<0.0> : tensor<1x1x1024x256xf32>} : () -> tensor<1x1x1024x256xf32>
%v463 = "tf.Conv2D"(%v458, %v462) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
%v460 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v461 = "tf.BiasAdd"(%v463, %v460) {data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
%v459 = "tf.Identity"(%v461) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v464 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v465 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v466 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v467 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v468, %v469, %v470, %v471, %v472, %v473 = "tf.FusedBatchNormV3"(%v459, %v464, %v465, %v466, %v467) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v474 = "tf.Relu"(%v468) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v478 = "tf.Const"() {value = dense<0.0> : tensor<3x3x256x256xf32>} : () -> tensor<3x3x256x256xf32>
%v479 = "tf.Conv2D"(%v474, %v478) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
%v476 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v477 = "tf.BiasAdd"(%v479, %v476) {data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
%v475 = "tf.Identity"(%v477) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v480 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v481 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v482 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v483 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v484, %v485, %v486, %v487, %v488, %v489 = "tf.FusedBatchNormV3"(%v475, %v480, %v481, %v482, %v483) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v490 = "tf.Relu"(%v484) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v494 = "tf.Const"() {value = dense<0.0> : tensor<1x1x256x1024xf32>} : () -> tensor<1x1x256x1024xf32>
%v495 = "tf.Conv2D"(%v490, %v494) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
%v492 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v493 = "tf.BiasAdd"(%v495, %v492) {data_format = "NHWC"} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
%v491 = "tf.Identity"(%v493) {} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v496 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v497 = "tf.Const"() {value = dense<1.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v498 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v499 = "tf.Const"() {value = dense<1.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v500, %v501, %v502, %v503, %v504, %v505 = "tf.FusedBatchNormV3"(%v491, %v496, %v497, %v498, %v499) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<*xf32>)
%v506 = "tf.AddV2"(%v458, %v500) {} : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v507 = "tf.Relu"(%v506) {} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v511 = "tf.Const"() {value = dense<0.0> : tensor<1x1x1024x256xf32>} : () -> tensor<1x1x1024x256xf32>
%v512 = "tf.Conv2D"(%v507, %v511) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
%v509 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v510 = "tf.BiasAdd"(%v512, %v509) {data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
%v508 = "tf.Identity"(%v510) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v513 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v514 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v515 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v516 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v517, %v518, %v519, %v520, %v521, %v522 = "tf.FusedBatchNormV3"(%v508, %v513, %v514, %v515, %v516) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v523 = "tf.Relu"(%v517) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v527 = "tf.Const"() {value = dense<0.0> : tensor<3x3x256x256xf32>} : () -> tensor<3x3x256x256xf32>
%v528 = "tf.Conv2D"(%v523, %v527) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
%v525 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v526 = "tf.BiasAdd"(%v528, %v525) {data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
%v524 = "tf.Identity"(%v526) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v529 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v530 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v531 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v532 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v533, %v534, %v535, %v536, %v537, %v538 = "tf.FusedBatchNormV3"(%v524, %v529, %v530, %v531, %v532) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v539 = "tf.Relu"(%v533) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v543 = "tf.Const"() {value = dense<0.0> : tensor<1x1x256x1024xf32>} : () -> tensor<1x1x256x1024xf32>
%v544 = "tf.Conv2D"(%v539, %v543) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
%v541 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v542 = "tf.BiasAdd"(%v544, %v541) {data_format = "NHWC"} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
%v540 = "tf.Identity"(%v542) {} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v545 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v546 = "tf.Const"() {value = dense<1.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v547 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v548 = "tf.Const"() {value = dense<1.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v549, %v550, %v551, %v552, %v553, %v554 = "tf.FusedBatchNormV3"(%v540, %v545, %v546, %v547, %v548) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<*xf32>)
%v555 = "tf.AddV2"(%v507, %v549) {} : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v556 = "tf.Relu"(%v555) {} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v560 = "tf.Const"() {value = dense<0.0> : tensor<1x1x1024x256xf32>} : () -> tensor<1x1x1024x256xf32>
%v561 = "tf.Conv2D"(%v556, %v560) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
%v558 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v559 = "tf.BiasAdd"(%v561, %v558) {data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
%v557 = "tf.Identity"(%v559) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v562 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v563 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v564 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v565 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v566, %v567, %v568, %v569, %v570, %v571 = "tf.FusedBatchNormV3"(%v557, %v562, %v563, %v564, %v565) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v572 = "tf.Relu"(%v566) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v576 = "tf.Const"() {value = dense<0.0> : tensor<3x3x256x256xf32>} : () -> tensor<3x3x256x256xf32>
%v577 = "tf.Conv2D"(%v572, %v576) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
%v574 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v575 = "tf.BiasAdd"(%v577, %v574) {data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
%v573 = "tf.Identity"(%v575) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v578 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v579 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v580 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v581 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v582, %v583, %v584, %v585, %v586, %v587 = "tf.FusedBatchNormV3"(%v573, %v578, %v579, %v580, %v581) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v588 = "tf.Relu"(%v582) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v592 = "tf.Const"() {value = dense<0.0> : tensor<1x1x256x1024xf32>} : () -> tensor<1x1x256x1024xf32>
%v593 = "tf.Conv2D"(%v588, %v592) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
%v590 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v591 = "tf.BiasAdd"(%v593, %v590) {data_format = "NHWC"} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
%v589 = "tf.Identity"(%v591) {} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v594 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v595 = "tf.Const"() {value = dense<1.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v596 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v597 = "tf.Const"() {value = dense<1.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v598, %v599, %v600, %v601, %v602, %v603 = "tf.FusedBatchNormV3"(%v589, %v594, %v595, %v596, %v597) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<*xf32>)
%v604 = "tf.AddV2"(%v556, %v598) {} : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v605 = "tf.Relu"(%v604) {} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v609 = "tf.Const"() {value = dense<0.0> : tensor<1x1x1024x256xf32>} : () -> tensor<1x1x1024x256xf32>
%v610 = "tf.Conv2D"(%v605, %v609) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
%v607 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v608 = "tf.BiasAdd"(%v610, %v607) {data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
%v606 = "tf.Identity"(%v608) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v611 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v612 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v613 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v614 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v615, %v616, %v617, %v618, %v619, %v620 = "tf.FusedBatchNormV3"(%v606, %v611, %v612, %v613, %v614) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v621 = "tf.Relu"(%v615) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v625 = "tf.Const"() {value = dense<0.0> : tensor<3x3x256x256xf32>} : () -> tensor<3x3x256x256xf32>
%v626 = "tf.Conv2D"(%v621, %v625) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
%v623 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v624 = "tf.BiasAdd"(%v626, %v623) {data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
%v622 = "tf.Identity"(%v624) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v627 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v628 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v629 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v630 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v631, %v632, %v633, %v634, %v635, %v636 = "tf.FusedBatchNormV3"(%v622, %v627, %v628, %v629, %v630) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v637 = "tf.Relu"(%v631) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v641 = "tf.Const"() {value = dense<0.0> : tensor<1x1x256x1024xf32>} : () -> tensor<1x1x256x1024xf32>
%v642 = "tf.Conv2D"(%v637, %v641) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
%v639 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v640 = "tf.BiasAdd"(%v642, %v639) {data_format = "NHWC"} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
%v638 = "tf.Identity"(%v640) {} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v643 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v644 = "tf.Const"() {value = dense<1.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v645 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v646 = "tf.Const"() {value = dense<1.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v647, %v648, %v649, %v650, %v651, %v652 = "tf.FusedBatchNormV3"(%v638, %v643, %v644, %v645, %v646) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<*xf32>)
%v653 = "tf.AddV2"(%v605, %v647) {} : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v654 = "tf.Relu"(%v653) {} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v658 = "tf.Const"() {value = dense<0.0> : tensor<1x1x1024x256xf32>} : () -> tensor<1x1x1024x256xf32>
%v659 = "tf.Conv2D"(%v654, %v658) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
%v656 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v657 = "tf.BiasAdd"(%v659, %v656) {data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
%v655 = "tf.Identity"(%v657) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v660 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v661 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v662 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v663 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v664, %v665, %v666, %v667, %v668, %v669 = "tf.FusedBatchNormV3"(%v655, %v660, %v661, %v662, %v663) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v670 = "tf.Relu"(%v664) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v674 = "tf.Const"() {value = dense<0.0> : tensor<3x3x256x256xf32>} : () -> tensor<3x3x256x256xf32>
%v675 = "tf.Conv2D"(%v670, %v674) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
%v672 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v673 = "tf.BiasAdd"(%v675, %v672) {data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
%v671 = "tf.Identity"(%v673) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v676 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v677 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v678 = "tf.Const"() {value = dense<0.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v679 = "tf.Const"() {value = dense<1.0> : tensor<256xf32>} : () -> tensor<256xf32>
%v680, %v681, %v682, %v683, %v684, %v685 = "tf.FusedBatchNormV3"(%v671, %v676, %v677, %v678, %v679) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)
%v686 = "tf.Relu"(%v680) {} : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
%v690 = "tf.Const"() {value = dense<0.0> : tensor<1x1x256x1024xf32>} : () -> tensor<1x1x256x1024xf32>
%v691 = "tf.Conv2D"(%v686, %v690) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
%v688 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v689 = "tf.BiasAdd"(%v691, %v688) {data_format = "NHWC"} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
%v687 = "tf.Identity"(%v689) {} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v692 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v693 = "tf.Const"() {value = dense<1.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v694 = "tf.Const"() {value = dense<0.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v695 = "tf.Const"() {value = dense<1.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
%v696, %v697, %v698, %v699, %v700, %v701 = "tf.FusedBatchNormV3"(%v687, %v692, %v693, %v694, %v695) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<*xf32>)
%v702 = "tf.AddV2"(%v654, %v696) {} : (tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v703 = "tf.Relu"(%v702) {} : (tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
%v707 = "tf.Const"() {value = dense<0.0> : tensor<1x1x1024x2048xf32>} : () -> tensor<1x1x1024x2048xf32>
%v708 = "tf.Conv2D"(%v703, %v707) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x2048xf32>) -> tensor<1x7x7x2048xf32>
%v705 = "tf.Const"() {value = dense<0.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v706 = "tf.BiasAdd"(%v708, %v705) {data_format = "NHWC"} : (tensor<1x7x7x2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
%v704 = "tf.Identity"(%v706) {} : (tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
%v709 = "tf.Const"() {value = dense<0.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v710 = "tf.Const"() {value = dense<1.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v711 = "tf.Const"() {value = dense<0.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v712 = "tf.Const"() {value = dense<1.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v713, %v714, %v715, %v716, %v717, %v718 = "tf.FusedBatchNormV3"(%v704, %v709, %v710, %v711, %v712) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> (tensor<1x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<*xf32>)
%v722 = "tf.Const"() {value = dense<0.0> : tensor<1x1x1024x512xf32>} : () -> tensor<1x1x1024x512xf32>
%v723 = "tf.Conv2D"(%v703, %v722) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x512xf32>) -> tensor<1x7x7x512xf32>
%v720 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v721 = "tf.BiasAdd"(%v723, %v720) {data_format = "NHWC"} : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
%v719 = "tf.Identity"(%v721) {} : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
%v724 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v725 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v726 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v727 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v728, %v729, %v730, %v731, %v732, %v733 = "tf.FusedBatchNormV3"(%v719, %v724, %v725, %v726, %v727) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<*xf32>)
%v734 = "tf.Relu"(%v728) {} : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
%v738 = "tf.Const"() {value = dense<0.0> : tensor<3x3x512x512xf32>} : () -> tensor<3x3x512x512xf32>
%v739 = "tf.Conv2D"(%v734, %v738) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
%v736 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v737 = "tf.BiasAdd"(%v739, %v736) {data_format = "NHWC"} : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
%v735 = "tf.Identity"(%v737) {} : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
%v740 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v741 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v742 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v743 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v744, %v745, %v746, %v747, %v748, %v749 = "tf.FusedBatchNormV3"(%v735, %v740, %v741, %v742, %v743) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<*xf32>)
%v750 = "tf.Relu"(%v744) {} : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
%v754 = "tf.Const"() {value = dense<0.0> : tensor<1x1x512x2048xf32>} : () -> tensor<1x1x512x2048xf32>
%v755 = "tf.Conv2D"(%v750, %v754) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) -> tensor<1x7x7x2048xf32>
%v752 = "tf.Const"() {value = dense<0.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v753 = "tf.BiasAdd"(%v755, %v752) {data_format = "NHWC"} : (tensor<1x7x7x2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
%v751 = "tf.Identity"(%v753) {} : (tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
%v756 = "tf.Const"() {value = dense<0.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v757 = "tf.Const"() {value = dense<1.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v758 = "tf.Const"() {value = dense<0.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v759 = "tf.Const"() {value = dense<1.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v760, %v761, %v762, %v763, %v764, %v765 = "tf.FusedBatchNormV3"(%v751, %v756, %v757, %v758, %v759) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> (tensor<1x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<*xf32>)
%v766 = "tf.AddV2"(%v713, %v760) {} : (tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
%v767 = "tf.Relu"(%v766) {} : (tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
%v771 = "tf.Const"() {value = dense<0.0> : tensor<1x1x2048x512xf32>} : () -> tensor<1x1x2048x512xf32>
%v772 = "tf.Conv2D"(%v767, %v771) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32>) -> tensor<1x7x7x512xf32>
%v769 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v770 = "tf.BiasAdd"(%v772, %v769) {data_format = "NHWC"} : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
%v768 = "tf.Identity"(%v770) {} : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
%v773 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v774 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v775 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v776 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v777, %v778, %v779, %v780, %v781, %v782 = "tf.FusedBatchNormV3"(%v768, %v773, %v774, %v775, %v776) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<*xf32>)
%v783 = "tf.Relu"(%v777) {} : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
%v787 = "tf.Const"() {value = dense<0.0> : tensor<3x3x512x512xf32>} : () -> tensor<3x3x512x512xf32>
%v788 = "tf.Conv2D"(%v783, %v787) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
%v785 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v786 = "tf.BiasAdd"(%v788, %v785) {data_format = "NHWC"} : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
%v784 = "tf.Identity"(%v786) {} : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
%v789 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v790 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v791 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v792 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v793, %v794, %v795, %v796, %v797, %v798 = "tf.FusedBatchNormV3"(%v784, %v789, %v790, %v791, %v792) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<*xf32>)
%v799 = "tf.Relu"(%v793) {} : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
%v803 = "tf.Const"() {value = dense<0.0> : tensor<1x1x512x2048xf32>} : () -> tensor<1x1x512x2048xf32>
%v804 = "tf.Conv2D"(%v799, %v803) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) -> tensor<1x7x7x2048xf32>
%v801 = "tf.Const"() {value = dense<0.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v802 = "tf.BiasAdd"(%v804, %v801) {data_format = "NHWC"} : (tensor<1x7x7x2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
%v800 = "tf.Identity"(%v802) {} : (tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
%v805 = "tf.Const"() {value = dense<0.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v806 = "tf.Const"() {value = dense<1.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v807 = "tf.Const"() {value = dense<0.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v808 = "tf.Const"() {value = dense<1.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v809, %v810, %v811, %v812, %v813, %v814 = "tf.FusedBatchNormV3"(%v800, %v805, %v806, %v807, %v808) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> (tensor<1x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<*xf32>)
%v815 = "tf.AddV2"(%v767, %v809) {} : (tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
%v816 = "tf.Relu"(%v815) {} : (tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
%v820 = "tf.Const"() {value = dense<0.0> : tensor<1x1x2048x512xf32>} : () -> tensor<1x1x2048x512xf32>
%v821 = "tf.Conv2D"(%v816, %v820) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32>) -> tensor<1x7x7x512xf32>
%v818 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v819 = "tf.BiasAdd"(%v821, %v818) {data_format = "NHWC"} : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
%v817 = "tf.Identity"(%v819) {} : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
%v822 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v823 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v824 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v825 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v826, %v827, %v828, %v829, %v830, %v831 = "tf.FusedBatchNormV3"(%v817, %v822, %v823, %v824, %v825) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<*xf32>)
%v832 = "tf.Relu"(%v826) {} : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
%v836 = "tf.Const"() {value = dense<0.0> : tensor<3x3x512x512xf32>} : () -> tensor<3x3x512x512xf32>
%v837 = "tf.Conv2D"(%v832, %v836) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
%v834 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v835 = "tf.BiasAdd"(%v837, %v834) {data_format = "NHWC"} : (tensor<1x7x7x512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
%v833 = "tf.Identity"(%v835) {} : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
%v838 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v839 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v840 = "tf.Const"() {value = dense<0.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v841 = "tf.Const"() {value = dense<1.0> : tensor<512xf32>} : () -> tensor<512xf32>
%v842, %v843, %v844, %v845, %v846, %v847 = "tf.FusedBatchNormV3"(%v833, %v838, %v839, %v840, %v841) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<*xf32>)
%v848 = "tf.Relu"(%v842) {} : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
%v852 = "tf.Const"() {value = dense<0.0> : tensor<1x1x512x2048xf32>} : () -> tensor<1x1x512x2048xf32>
%v853 = "tf.Conv2D"(%v848, %v852) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true, data_format = "NHWC"} : (tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) -> tensor<1x7x7x2048xf32>
%v850 = "tf.Const"() {value = dense<0.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v851 = "tf.BiasAdd"(%v853, %v850) {data_format = "NHWC"} : (tensor<1x7x7x2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
%v849 = "tf.Identity"(%v851) {} : (tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
%v854 = "tf.Const"() {value = dense<0.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v855 = "tf.Const"() {value = dense<1.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v856 = "tf.Const"() {value = dense<0.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v857 = "tf.Const"() {value = dense<1.0> : tensor<2048xf32>} : () -> tensor<2048xf32>
%v858, %v859, %v860, %v861, %v862, %v863 = "tf.FusedBatchNormV3"(%v849, %v854, %v855, %v856, %v857) {data_format = "NHWC", epsilon = 1.001e-05 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<1x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> (tensor<1x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<*xf32>)
%v864 = "tf.AddV2"(%v816, %v858) {} : (tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
%v865 = "tf.Relu"(%v864) {} : (tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
%v866 = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
%v867 = "tf.Mean"(%v865, %v866) {data_format = "NHWC", keep_dims = false} : (tensor<1x7x7x2048xf32>, tensor<2xi32>) -> tensor<1x2048xf32>
%v868 = "tf.Const"() {value = dense<0.0> : tensor<2048x1000xf32>} : () -> tensor<2048x1000xf32>
%v869 = "tf.MatMul"(%v867, %v868) {transpose_a = false, transpose_b = false} : (tensor<1x2048xf32>, tensor<2048x1000xf32>) -> tensor<1x1000xf32>
%v871 = "tf.Const"() {value = dense<0.0> : tensor<1000xf32>} : () -> tensor<1000xf32>
%v872 = "tf.BiasAdd"(%v869, %v871) {data_format = "NHWC"} : (tensor<1x1000xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
%v873 = "tf.Softmax"(%v872) {} : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
lus.yield(%v873: tensor<1x1000xf32>)
}

