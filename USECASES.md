### Common configuration

If needed, change variables at the beginning of the Makefile. 

Example : variable
```MLIRLUS ``` must be set to the path of your build of ```mlir-lus```
(normally ```../../mlir-lus```).

### Pitch tuning vocoder

### Build the use case

Move to the ```pitch-sched``` directory and build the use case ```make```
The result of compilation is file ```pitch```.

### Running the use case

To allow execution, the computer must have a sound input and 
a sound output device. On a common notebook, these are either 
the built-in microphone and speakers, or a headset. The second
variant is recommended, as it avoids sound feed-back (Larsen
effect).

It is assumed that SoX has been configured to use these as
default input and output devices. This can be tested by running 
the following command, which should result in an echo:
```
   rec -q -t raw -r 44100 -e signed -b 16 -c 2 - | \
                play -q -t raw -r 44100 -e signed -b 16 -c 2 -
```

To launch execution, execute ```make run```.  The application will
continuously capture the input from the standard sound device and send
the processed output to the standard sound device. When you talk into
the microphone, the output will be your voice with a changed pitch (by
default, higher by 3 semitones). If the result is an increasing noise,
it is probable that you do not have a headset, and that a feedback
(Larsen effect) takes place. The solution is to reduce the system
volume (or use a headset).

To stop execution, press ```<CTRL>-C```, followed by ```make kill```.

During execution, one can change:
* The volume, by 10% increments, using keys ```+``` and ```-```
      (one positive or negative increment by key).
* The pitch shift, by half-semitone increments, using
      keys ```n``` (positive increment) or ```m``` (negative increment)
* Mute the sound using key ```s```
* Unmute the sound using key ```a```

To provide a command key, write it, and then press ```<ENTER>```.
    Multiple command keys can be placed on a single line
    before pressing ```<ENTER>```. For instance, writing:

	mmmmmmm<ENTER>
		
will reduce the pitch by 3.5 semitones.
   
We use this (primitive) interface in order to avoid further
    reliance on external libraries such as ncurses.

### Description of the use case

The source code of the use case consists of files of 3 types:
* ```lus``` dialect code is grouped in file ```pitch.lus.mlir```, 
  where it is mixed with regular MLIR code of the ```Standard```
  dialect. 

* Regular MLIR code is grouped in the other ```*.mlir``` files.
  This comprises the majority of the use case. It can be 
  integrated with the reactive code, but we preferred keeping
  the reactive code apart for lisibility.
  
* C code:
  * The implementation of the longjmp-based scheduler.
  * ```sndio.c``` provides the soundcard read and write routines,
    which cannot be written in MLIR in the standard dialect (only
	in the lower-level LLVM dialect).
  * ```bitrev.c``` is the only only function written in C
    that does not interface with the OS. It can be
    converted to MLIR, but we preferred showcasing the ability 
	to interface.

## Resnet50

Resnet50 is a convolutional neural network which is 50 layers deep which
performs computationnally-intensive operations.

### Resnet50 as our usecase

The directories usecases/resnet50 (on which we performed our benchmarks and 
which splits lus and tensorflow code in two files) and
usecases/resnet50-compact (which provide an implementation combining lus and
tensorflow code in the same file) contain Resnet50 as our usecases.

### Build the use case

Move to the ```resnet50``` directory and build the use case ```make```
The result of compilation is file ```resnet```. It prints the timestamp at
each cycle.

### Production of Resnet50

We provide a full (trained) Resnet50 implementation, but you can produce it 
yourself, from the Python Tensorflow library. You have to launch the 
usecases/produce-resnet50/produce-resnet50.sh script. In the produced 
resnet.mlir file, you just have to :
- Manually remove the wrapping operations tf\_executor.graph and 
  tf\_executor.island
- Manually replace the value produced by the "tf.Placeholder" op by a function
  parameter.
- Manually return the last value produced.
- Update the function type.

Then you can use this file instead of usecases/resnet50/resnet.tf.mlir.
If you (manually) change the function into a lus node, you can use it instead 
of usecases/resnet50-compact/resnet.lus.mlir.

## Timeseries

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


