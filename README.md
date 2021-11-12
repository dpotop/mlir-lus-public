# Reactive programming in [MLIR](https://mlir.llvm.org)

While compilation frameworks such as [MLIR](https://mlir.llvm.org) concentrate the existing know-how in
HPC compilation for virtually every execution platform, they lack a key ingredient needed
in the high-performance embedded systems of the future: the ability to represent reactive
control and real-time aspects of a system. They do not provide first-class representation and
reasoning for systems with a cyclic execution model, synchronization with external time
references (logical or physical), synchronization with other systems, tasks and I/O with
multiple periods and execution modes. In practice, this poses problems when representing 
TensorFlow control in the [tf_executor dialect](http://download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf), when providing streaming implementations for
ML applications, when integrating ML components with signal processing pipelines...

We propose two extensions of MLIR dedicated to reactive programming. Following MLIR convention, these
extensions come under the form of so-called [dialects](https://mlir.llvm.org/docs/LangRef/#dialects):
* **lus**  is a high-level reactive programming dialect based on 
           [Lustre](https://en.wikipedia.org/wiki/Lustre_(programming_language)). Specification of application
	   control at lus dialect level comes with formal correctness checking ensuring the absence of 
	   infinite or undefined behaviors. In turn, this ensures that embedded implementation is possible 
	   in bounded memory space and in bounded execution time. The lus dialect follows a dataflow programming
	   paradigm allowing the natural specification of ML specifications with complex control and 
	   integration with signal processing pipelines.
* **sync** is a low-level reactive programming dialect. It directly extends the [Static 
           Single Assignment (SSA)](https://en.wikipedia.org/wiki/Static_single_assignment_form) form 
	   which stands at the core of MLIR with reactive primitives allowing synchronization with 
	   other functions and with external time references, I/O with multiple periods and execution modes.
	  
Both dialects freely combine with the data processing operations already present in MLIR
(in dialects such as [tensorflow]() or [linalg]()), thus allowing joint specification of
all aspects of an embedded system - high-performance data processing and interaction with the
environment.

Our current effort is focused in two directions:
* Promoting the lus dialect for the specification of high-performance embedded applications featuring ML and signal processing under complex reactive control.
* Improving code generation.

In developing our dialects and their compilation process, we follow the maximal reuse approach encouraged by MLIR (even when this approach comes with a steeper learning curve).

Further reading: [H. Pompougnac, U. Beaugnon, A. Cohen, D. Potop - From SSA to Synchronous Concurrency and Back](https://hal.inria.fr/hal-03043623/document)

# Getting started

The repository comprises 3 main folders:
* **mlir-lus** contains the implementation of the lus and sync dialects, and of the command-line **mlirlus*** tool allowing compilation of specifications based on these dialects. 
* **mlir-prime** is a tool that exposes existing MLIR code transformations that are not exposed by the command-line transformation tools of MLIR. 
* **usecases** showcases the use of our new dialects on a few signal processing and ML applications.

## [Installation instructions](INSTALL.md)

## [Invocation of mlirlus and mlirprime](INVOCATION.md)

## [Compiling and executing the use cases](usecases/README.md)

# Licensing

TODO - for now there is no license, meaning that code cannot be reused. We are currently evaluating our options.

# Contact information

The authors of mlirlus are Hugo Pompougnac and [Dumitru Potop](https://github.com/dpotop).
