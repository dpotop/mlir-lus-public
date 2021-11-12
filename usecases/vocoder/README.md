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
