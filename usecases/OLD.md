### Pitch tuning vocoder

#### Build the use case

Move to the ```pitch-sched``` directory.

1- If needed, change variables on top of the Makefile. Variable ```LLVMDIR``` 
   must be set to the installation folder of ```llvm``` 
   (normally ```$(HOME)/llvm```). Variable ```MLIRLUS ``` must be set to the 
   path of your build of ```mlir-lus``` (normally ```../mlir-lus```).

2- Build the use case in the pitch-lus-threads directory with ```make```

The result of compilation is file ```pitch```.

#### Running the use case

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

#### Description of the use case

The source code of the use case consists of files of 3 types:
* ```lus``` dialect code is grouped in file ```pitch.lus```, 
  where it is mixed with regular MLIR code of the ```Standard```
  dialect. 

* Regular MLIR code is grouped in the ```*.mlir``` files.
  This comprises the majority of the use case. It can be 
  integrated with the reactive code, but we preferred keeping
  the reactive code apart for lisibility.
  
* C code:
  * TODO scheduler
  * ```sndio.c``` provides the soundcard read and write routines,
    which cannot be written in MLIR in the standard dialect (only
	in the lower-level LLVM dialect).
  * ```bitrev.c``` is the only only function written in C
    that does not interface with the OS. It can be
    converted to MLIR, but we preferred showcasing the ability 
	to interface.

#### The reactive code and its compilation

All reactive code is contained in file ```pitch.lus``` and 
consists of standard MLIR and ```lus``` dialect code.
The code is divided in 3 nodes:
* Node ```@pitch``` implements the top-level I/O and control,
  including the sliding window over input samples, and the
  mute/unmute control. It
  instantiates nodes ```@pitch_algo``` and ```@kbd_ctrl```
* Node ```@pitch_algo``` implements the pitch shifting
  signal processing algorithm.
* Node ```@kbd_ctrl``` implements the keyboard interaction
   algorithm that changes the pitch correction configuration.	
	
NOTE: In the source code, all ```lus``` and ```sync``` dialect 
operations are prefixed with the dialect name. Thus, ```lus.fby``` 
is the operation ```fby``` of dialect ```lus```.

Our reactive source code showcases all features of the Lustre 
language described in the paper:
* Cyclic execution model (implicit)
* Cyclic I/O through the interface variables of the three nodes.
* Feedback (program state) implemented using the ```lus.fby```
  operation.
* Dataflow expression of conditional execution, using the 
  ```lus.when``` and ```lus.merge``` operations.
* Modularity.

The first step in the compilation of this code is to lower
it to the abstraction level of the ```sync``` dialect
described in Section 3 of the submitted paper. This 
is done through the command:

```../mlir-lus/mlirlus --normalize --convert-lus-to-sync pitch.lus```

At this level:
* State is encoded using existing mechanisms provided by standard SSA.
  In our paper, this is done using variables and phi operations. 
  The output of our tool uses higher-level structured control 
  flow constructs provided by MLIR (as part of the ```SCF``` dialect),
  which facilitate subsequent memory allocation phases (but will 
  ultimately be lowered to standard dialect).
* Variable-based inputs and outputs have been transformed into 
  I/O channel variables and ```sync.input``` and ```sync.output``` operations.
* The implicit cyclic execution model, implicit activation conditions,
  and implicit cycle separation is made explicit using standard 
  SSA control flow and the ```sync.tick``` operation. This operation also
  requires the explicit use of ```sync.undef``` operations.

From this level, a further lowering phase, materialized in the 
command line option ```--convert-sync-to-std``` will lower 
the ```sync``` dialect operations to standard dialect MLIR, 
including calls to library functions such as ```tick()```, 
defined below. 

At this point, no memory allocation has yet been performed. The 
compilation script does that using the ```mlir-opt``` tool 
of MLIR, as described in the Makefile.

### Time Series application

TODO - add description

### Resnet50

TODO - add description
TODO - add instructions on importing the example from Keras, or the 
       code with constant weights (and specify it).
