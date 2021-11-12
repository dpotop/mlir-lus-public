## mlirlus - command line options
Compilation produces the command line tool mlirlus.
It takes one input file and prints its output
on the standard output. 

The command : ```./mlirlus file.lus``` just parses
file.lus, verifies its syntactic correction and prints it unchanged.

mlirlus takes several options, which activate the various Lustre-specific
compilation steps described in [this paper](https://hal.inria.fr/hal-03043623/document):
* ```--normalize``` performes the lus dialect normalization step.
  It makes all fby and pre operations work on the base clock (at every cycle)
  and then moves their representation in the node signature, as loop-carried dependences. 
  This ensures the respect of dominance rules. 
* ```--inline-nodes```. When requested using the ```inline``` keyword in the program
  text, this step inlines instantiated nodes to avoid creating a new thread
  in the implementation.
* ```--convert-lus-to-sync``` converts lus operations to a combination of 
  operations of the dialects sync and scf (structured control flow).
* ```--convert-sync-to-scf``` converts sync operations to std operations,
  including calls to the run-time functions implementing the reactive 
  execution machine.

## mlirprime - command line options

Recall that mlirprime is an auxiliary tool which exposes a number of MLIR code
transformations (and corrects some bugs).

The tool takes an input file and prints its output
on the standard output. The command : ```./mlir-prime file.prime``` just parses
file.prime, verify its correction and prints it.

Several options can be composed with the builtin options of MLIR :
* ```--bufferize-linalg-reshape``` corrects a limitation of the bufferization
  passes on the version of MLIR we used for the use cases (this limitation
  is corrected in the current MLIR version).
* ```--remove-copy-prime``` removes linalg.copy operations on which we had
  problems for lowering.
* ```--prime-linalg-to-affine``` bypasses the builtin lowering of the dialect 
  linalg to the dialect affine using packing, tiling... These optimisations
  are machine-dependent and correspond to our benchmark architecture.
* ```--loop-permutation-prime``` extends the built-in ```loop-permutation```
  option which just operates on the first loop of the program. We are
  currently submitting this extension to the MLIR project.
