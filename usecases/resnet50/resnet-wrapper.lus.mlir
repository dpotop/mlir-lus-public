func private @resnet(tensor<1x224x224x3xf32>) -> (tensor<1x1000xf32>)
lus.node @wrapper(%in: tensor<1x224x224x3xf32>) -> (tensor<1x1000xf32>) {
  %out = call @resnet(%in): (tensor<1x224x224x3xf32>) -> (tensor<1x1000xf32>)
  lus.yield(%out: tensor<1x1000xf32>)
}
