#### Table of Contents  
* [The ```lus``` dialect](#lus)
* [The ```pssa``` dialect](#pssa)
* [Command line options](#general_options)
  * [Input and output files](#io)
  * [Processing pipeline control](#processing)
  * [Debugging options](#debugging)
* [Compilation and linking](#compilation)  
  * [Interfacing MLIR and external (C) code](#interfacing)
  * [Debugging](#debugging)
  * [Bugs](#bugs)
  * [TODO](#todo)



# The ```lus``` dialect <a name="lus"/> 

## Syntax

The dialect defines no new types, but its operations can use values of
any type of another dialect (unless this other dialect restricts the use).

* ```lus.node``` has two forms:
  * ```lus @name [static(named_value_list)] [state(named_value_list)] (named_value_list) -> (type_list)``` 
  This form only declares a node signature with the given 
  static, state, input and output arguments, respecively. 
  Its form is by no means perfect, but MLIR places some 
  limits on syntactic creativity
  (no <>, using [] is severely curtailed...). I could probably do 
  better, with smth like ((/)) and (((/))) list bounds for 
  states and static arguments, but this demands some thinking.
  The list of static arguments is optional. Is absence means 
  that there are no state arguments, but the same effect can be 
  achieved if there are no arguments between parentheses.
  The list of state arguments is also optional. Its absence 
  either means that the node has no state, or that the state 
  is exclusively expressed using ```fby``` and ```pre``` 
  operations, and instantiated node states. A normalization 
  phase pushes all these internally-handled states into the 
  list of state arguments, allowing code generation.
  The input and output lists must always be present, but 
  they can be empty. 
  * ```lus [dom] @name [static(named_value_list)] [state (named_value_list)] (named_value_list) -> (type_list) { operation_list }``` 
  This form also provides the implementation of the node in the 
  form of a single region containing the list of operations of the node. 
  The optional ```dom``` flag specifies that the operations of the region 
  are topologically ordered, satisfying the dominance property as checked
  by MLIR. This activates the dominance check of the MLIR region, as 
  part of the correctness verifications.
* ```lus.yield [state (named_value_list)] (named_value_list)``` This is the 
only accepted terminator of a ```lus.node``` region. The number and type 
of state and output values must correspond to the number of type of elements 
in the state and output lists in the node signature. Important point: 
the state of instantiated nodes, as well as that of ```pre``` and ```fby``` 
operations is not transmitted through the state variables here.
* ```value = lus.fby value value : type``` Its is required that the 3 
values have the the same data type and same clock. The first input value 
is used only at the first cycle. The second value is the one that's 
delayed. Note that the current syntax does not work on tuples. Furthermore,
MLIR does not offer in the standard dialect(s) tuple construction or tuple 
access operations, meaning that I should define them myself, if I really 
feel the need for it. Recall, however, that this is supposed to be an IR,
meaning that there's no problem in keeping separate values of a tuple
distinct. The only possible issue concerns clocking, and that can be solved
with dedicated function-like operations that impose clock equality and 
generate no code.
* ```value = lus.pre value : type``` The difference w.r.t. ```lus.fby``` 
is that the first cycle where the input value is present, the output 
value is not. Formally, if the clock of the input value is ```c```, 
that of the output value will be ```c when 0(1)```.
* ```value = lus.when clock value : type``` When ```clock``` is ```%v``` 
or ```not %v``` for some value ```%v``` of type ```i1```, the semantics 
is that of classical Lustre. But ```clock``` can also be a k-periodic 
word, in which case the semantics is that of n-synchronous sub-sampling.
For k-periodic words, ```not```cannot be used. Instead, the word must be
flipped. The type is that of the data values.
* ```value = lus.merge clock value value : type``` Classical merge. The 
clock can be either some value ```%v``` of type ```i1```, or a k-periodic
word. The type is that of the data values.
* ```[result =] lus.instance [noinline] @node_name (value_list) : (type_list) -> (type_list)``` 
This  operation allows the hierarchic dataflow composition. The result 
can be missing if there are no results. The optional ```noinline``` 
specifies that this instance is not to be inlined during inlining 
phases, which otherwise inline everything. The ```value_list```  and 
the first (input) ```type_list``` must have the same length. The node 
of name (symbol) ```node_name``` must exist and have the correct interface.
This syntax does not currently account for static arguments or state
arguments.

Types and operations of most other dialects can be used to represent 
data processing in a Lustre node. 
Among these, ```std.call```, ```std.call_indirect```,
and all operation of the ```std```dialect 
save ```std.func```, ```std.br```, ```std.cond_br```, ```std.yield```, ```std.assume_alignment```.

NOTE: static arguments of nodes are not yet fully implemented in the
parser (only in node interfaces, not in lus.instance), and are not 
implemented in the code generator. Same for the ```noinline``` flag 
of instances.

## Semantics

Overall, the semantics is that of the Lustre language, extended with
the data semantics of MLIR and with the particular quirks any implementation
has. 

### Single assignment 

Inside each ```node``` that is defined (i.e. has a region delimited by 
{}), each value must be uniquely defined, either by one operation, or as 
an input or state argument of the ```node```.

Furthermore, any region must be terminated by ```lus.yield```, even if 
there are no outputs or states it manipulates.

### Causality

The causality requirement is only verified on nodes carrying the ```dom```
flag. The verification is simply that dominance (as defined by MLIR) holds.

### Clock analysis

Clock analysis is performed on a per-node basis, on the nodes that have an 
implementation. For nodes that only have a declaration, it is assumed that
all inputs and outputs have the base clock of the node. This allows the 
absence of clock annotations on the node interface. Adding these clock 
annotations is under way.

Clock analysis follows the same principles as the one in Heptagon, but 
is



# The ```pssa``` dialect <a name="pssa"/>

The dialect defines one type ```pssa.pred``` used to represent predicates.
Through lowering, it becomes std.i1, but we preferred keeping it apart, as
its implementation is not necessarily a Boolean variable (e.g. in predicated
SSA approaches it can be mapped on special processor values).

* ```pssa.psi```
* ```pssa.condact```
* ```pssa.create_pred``` Two forms, one with a value as input, the other with
a k-periodic word.
* ```pssa.undef```



# Command line options of mlirlus <a name="general_options"/>

Example ```./mlirlus --normalize --predicate --inline-nodes --convert-lus-to-std -mainnode=test --convert-pssa-to-std --verbose=1 demo/clock_calculi.mlir -o demo/clock_calculi_std.mlir```



## Input and output files <a name="io"/>

* The input file is set using the first positional argument, which is either a file name or ```-```. The latter specifies that input is taken from the standard input, and is the default option.

* ```-o=name``` sets the output file. The name is either a regular file name or ```-```. The latter specifies that output is done on the standard output, and is the default value.



## Processing pipeline control <a name="processing"/>

* ```--inline-nodes``` Recursively inline all node instances that are not tagged with ```noinline```.

* ```--normalize``` Normalize lus nodes of an MLIR file. Normalization is performed on a per-node basis. All other definitions (e.g. functions of the std dialect) are left unchanged. Normalization involves a number of transforms:
  * Replacement of ```fby``` and ```pre``` operations with loop-carried dependences represented by ```lus.yield``` and the node interface.
  * Enforce the dominance property and set the ```dom``` flag on the node definition. When applied to a node without the dominance property, it will topologically order its operations in a way that is compatible with dominance (and it will set the ```dom``` flag). In nodes with the ```dom```flag, it does nothing, as dominance is checked at node load/verification. Topological ordering fails if the definitions are cyclic.
  * In the future, normalization must ensure that all node instances are replaced with function calls (or some futures), and that only one node is preserved for full-fledged global loop implementation (and this, only when it is meant for this role).

* ```--predicate``` Adds pssa dialect predication to a lus node. This makes predication (clocks) explicit.

* ```--convert-lus-to-std``` Lowers lus operations (including nodes) to the standard dialect. Assumes predicates have already been synthesized.

* ```--mainnode=name``` Upon non-modular lowering of lus nodes to std functions, sets the name of the node that will be lowered.

* ```--convert-pssa-to-std``` Lower pssa operations and types to the standard dialect. Assumes lus constructs have already been lowered to the std dialect.



## Debugging options <a name="debugging"/>

* ```--verbose=N``` Set the verbosity level, which is by default 0. The higher the level is, the more tracing/debugging information is printed on the console during the execution of mlirlusc.

* ```--disable-clock-analysis``` Disable clock analysis. As code generation depends on clock analysis, this can only be used in order to debug the parser.

* ```--show-dialects``` Print the list of registered dialects and exit.



# Compilation and linking <a name="compilation"/>

The tool compiles with the 09/23/2020 version of LLVM/MLIR 
(commit bd8b50cd7f5dd5237ec9187ef2fcea3adc15b61a).

TODO

## Interfacing MLIR and external (C) code, in both senses <a name="interfacing"/>

TODO



## Debugging and defensive programming <a name="debugging"/>

This section contains information on how to debug different aspects of 
MLIR and the lus and pssa dialects. Given that implementation is done in 
C++, the need for low-level debugging appears relatively fast, and any
meaningful change of the code can result in low-level (memory 
allocation) bugs.

Memory debugging is easily done using the AddressSanitizer facility, 
which is part of the clang distribution. Per 
[the LLVM AddessSanitizer documentation](https://clang.llvm.org/docs/AddressSanitizer.html),
using the facility amounts to including ```-fsanitize=address``` to both compilation and
linking. This is already done in the current Makefile. 

Note that, to do this on your MLIR code, you also have to add the flags 
when compiling LLVM/MLIR itself (unless your code is already included in MLIR, in which 
case you have to pass the caudine forks of code review). You can do this by changing 
the LLVM compilation command into something like: ```
cmake -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_INSTALL_PREFIX=/Users/dpotop/llvm -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang-mp-9.0 -DCMAKE_CXX_COMPILER=clang++-mp-9.0 -DCMAKE_ASM_COMPILER=clang-mp-9.0 -DCMAKE_CXX_FLAGS_DEBUG="-fno-omit-frame-pointer -fsanitize=address" -DCMAKE_LINKER_FLAGS_DEBUG="-fno-omit-frame-pointer -fsanitize=address" ../llvm
```

Doing this impacts both compilation speed and execution speed. 
According to documentation, execution speed is halved.

Once this is done, execution time memory errors will not be simple crashes.
Instead, a nice report will indicate what type of memory error there was, and
in which program line. Along with some printing code (or classical debugging, 
although doing it with lldb is quite different from the intuitive gdb) this is
quite enough for general problems.

## Known bugs <a name="bugs"/>

## TODO <a name="todo"/>
* Accept values from outside the scope - mainly global
* Modular code generation
  * To allow true modularity, without knowing the content of the 
    instantiated nodes, I need to pass opaque types. Named opaque types
	currently don't exist in MLIR. However, on the caller side I 
	can pass ```memref<i32>```, whereas on the callee side 
	it's ```memref<tuple<...>>```. The linker does the rest. 
	Of course, this requires each node to be in a separate object file.
	To do this efficiently, I may need to add to the node type a flag 
	determining if it has no state.
	BTW: currently, ```memref<tuple<...>>``` is not accepted by MLIR. I
	asked a question on the forum.
  * I could also make the assumption that I know everything about
    the called node, including it's full state structure. But is this
	realistic?
* Add type annotations on interface, as well as clock analysis that
  takes them into account. The syntax:
   
* Enforce memory allocation

