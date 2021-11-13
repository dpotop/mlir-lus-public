Resnet50 is a deep neural network. The compressed size of its weights
is over 100Mo, meaning that we cannot directly distribute it through
github. 

We provide the weights through our [Google Drive](https://drive.google.com/drive/folders/1ZzShwEBQZGZftc575yGGSwNdAzWtsRUe?usp=sharing). The folder contains one compressed file ```resnet.tf.mlir.gz```,
which must be downloaded and uncompressed inside the ```usecases/resnet50```
folder with the command ```gzip -d resnet.tf.mlir.gz```.

To compile, execute ```make```, which produces the coroutine-based 
executable file ```resnet```, or ```make sequential```, which
produces the executable file ```resnet-seq```. Execution of these files will
print the timestamp and duration at the end of each execution cycle.
