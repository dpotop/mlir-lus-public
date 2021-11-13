### Build the use case

To build the coroutine-based implementation, execute ```make```. To build the sequential implementation, execute ```make sequential```. The result of compilation is the executable file ```pitch```, for the coroutine-based implementation and 
```pitch-seq``` for the sequential one.

### Sound system configuration

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

If your sound system does not accept the 44100 Hz sampling frequency (error
message), you may need to change it to one that is accepted, e.g. 48000 Hz.
In this case, you should also update the frequency in the ```run```
rule of the Makefile. 

To determine the format natively accepted by your sound card, 
execute ```rec -t raw test.raw```. The call will print the detected
configuration, which must then be modified in the Makefile. The 
options that may need modification in the "rec" and "play" calls
are:
* -r = sampling rate
* -e = signed/unsigned
* -b = sample size (in bits)
* -c = number of channels (2 for stereo)

### Running the use case

To launch the execution of the coroutine-based implementation, execute ```make run```.  
To launch the execution of the sequential implementation, execute ```make run-seq```.

In both cases, the application will
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
      (one positive or negative increment by key press).
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
    reliance on external libraries such as ncurses, and thus 
    ensure maximal portability.

### Description of the use case

The source code of the use case consists of files of 3 types:
* ```lus``` dialect code is grouped in file ```pitch.lus.mlir```, 
  where it is mixed with regular MLIR code of the ```Standard```
  dialect. 

* Regular MLIR code is grouped in the other ```*.mlir``` files.
  This comprises the majority of the use case. It could be 
  integrated with the reactive code, but we preferred keeping
  the reactive code apart for lisibility.
  
* C code:
  * The implementation of the longjmp-based scheduler.
  * ```sndio.c``` provides the soundcard I/O routines 
    which use ```read``` and ```write``` system calls. This code
    cannot be written in MLIR in the standard dialect (due to 
    vector data representation issues).
  * ```bitrev.c``` is the only only function written in C
    that does not interface with the OS. It can be
    converted to MLIR, but we preferred showcasing the ability 
	to interface with legacy C code.
