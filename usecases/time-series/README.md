
This Recurrent Neural Network with one single LSTM layer is the one described
in the paper. We reimplemented it on the basis of the method used for Resnet50,
but it needed more significant modifications (about the steate representation).
We also reduce the learnt-weigths to zero constant tensors.

### Timeseries as our usecase

The directory usecases/time-series provides our implementation interfaced to
the longjmp-based scheduler, and the directory usecases/time-series-inlined
provides an (earlier) version that just inlines nodes insteaf of truly
instantiate them.

### Build the use case

Move to the ```time-series``` directory and build the use case ```make```
The result of compilation is file ```rnn```. It prints the timestamp at each
cycle.

