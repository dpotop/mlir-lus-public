// -*- C++ -*- //

#ifndef MLIRLUS_NODETMP_H
#define MLIRLUS_NODETMP_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
// Needed to include the RegionKindInterface and its trait
#include "mlir/IR/RegionKindInterface.h" 
#include "NodeType.h"
#include "Yield.h"

#include <iostream>
#include <assert.h>

namespace mlir {
  namespace lus {

    /// Dataflow nodes template class. Nodes allow the expression of
    /// both:
    /// - Nodes that do not enforce the dominance property. In such
    ///   nodes, the general use of the fby construct and of node
    ///   instantiation is allowed, which can create dominance cycles
    ///   as in:
    ///   > %w = lus.fby (%k :i32 %w : i32) : i32
    ///   > %e = lus.instance @mynode (%e): (i32) -> (i32)
    ///   Note that, in correct specifications, dominance cycles like
    ///   those above mean that the function of at least one operation in
    ///   the dataflow must be implemented by multiple pieces of code,
    ///   scheduled separately.
    ///   Absence of dominance also means that specs without cycles
    ///   can have their equations not following dominance ordered, e.g.
    ///   having the operation using a value before the operation
    ///   defining it.
    ///
    /// - Nodes where the dataflow definition respects dominance. Such
    ///   nodes can be directly presented to scheduling and code
    ///   generation.
    ///
    ///   Such nodes can still contain lus.fby and lus.instance
    ///   operations, but these must not create dominance cycles.
    ///   
    ///   Our compilation process, however, seeks to fully control
    ///   scheduling and memory allocation, and to exploit a maximum
    ///   freedom of scheduling (at some hierarchy level). For this
    ///   reason, transformations will:
    ///   * Transform all node instantiations into function calls with
    ///     explicit state representation using values and lus.fby
    ///     operations. Of course, modular code generation is also
    ///     possible, but it restricts freedom of scheduling by requiring
    ///     that computations and state updates are performed together.
    ///   * Move state elements represented using lus.fby into the
    ///     lus.yield form (fby cannot be generally represented in
    ///     a dominance-compliant way).
    ///
    /// As in MLIR dominance is handled as a trait -- a static property
    /// of a class, rather than a property of objects, I need two
    /// distinct operations to represent the two cases. However, they
    /// share a lot of functionality, and in particular the ability
    /// to trade off between:
    /// - Unmanaged state, present in fby operations and node instances.
    /// - Managed state, handled like loop-carried dependencies through
    ///   the lus.yield operation and the node interface.
    /// 
    /// For this shared functionality, we set up a template class.
    class NodeOp : public Op <
      NodeOp,
      SymbolOpInterface::Trait,
      OpTrait::ZeroOperands,
      OpTrait::ZeroResult,
      OpTrait::NRegions<2>::Impl,
      OpTrait::IsIsolatedFromAbove,
      OpTrait::ZeroSuccessor,
      CallableOpInterface::Trait,
      RegionKindInterface::Trait > {
      
    public:      
      using Op<NodeOp,
	       SymbolOpInterface::Trait,
	       OpTrait::ZeroOperands,
	       OpTrait::ZeroResult,
	       OpTrait::NRegions<2>::Impl,
	       OpTrait::IsIsolatedFromAbove,
	       OpTrait::ZeroSuccessor,
	       CallableOpInterface::Trait,
	       RegionKindInterface::Trait>::Op ;

    public:
      
      static StringRef getOperationName() { return "lus.node"; }
      StringRef getNodeName() ;
      static void build(Builder &builder, OperationState &state,
      			StringRef name,
      			ArrayRef<Value> statics,
      			ArrayRef<Value> inputs,
			ArrayRef<Value> states,
      			ArrayRef<Type> resultTypes,
      			RegionKind regionKind);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      LogicalResult verify();

      /// Type Handling

      NodeType getType() { return getTypeAttr().getValue().cast<NodeType>(); }

      /// Regions handling

      /// Required by RegionKindInterface
      static RegionKind getRegionKind(unsigned);
      /// Required by CallableOpInterface
      Region *getCallableRegion();
      /// Get the body if exists ; if not, fails
      Region& getBody();
      /// Check if the body exists
      bool hasBody() { return !getActiveRegion().empty() ; }
      /// Check witch region is active (ie if dom verification is shut off)
      bool isDominanceFree();
      /// Turn on dominance
      void forceDominance();

      /// Interface handling ; static, inputs, states and outputs fields

      unsigned getNumStatic() { return getType().getNumStatic() ; }
      unsigned getNumInputs() { return getType().getNumInputs() ; }
      unsigned getNumStates() { return getType().getNumStates() ; }
      unsigned getNumOutputs() { return getType().getNumResults() ; }

      ArrayRef<Type> getStaticTypes() { return getType().getStatics(); }
      ArrayRef<Type> getInputsTypes() { return getType().getInputs(); }
      ArrayRef<Type> getStatesTypes() { return getType().getStates(); }
      ArrayRef<Type> getCallableResults() { return getType().getResults();}
      ArrayRef<Type> getOutputsTypes() { return getCallableResults(); }

      iterator_range<Block::args_iterator> getStatics();
      iterator_range<Block::args_iterator> getInputs();
      iterator_range<Block::args_iterator> getStates();

      /// Add a new state variable to the node interface
      Value addState(Type type);

      YieldOp getYield();

    private:

      static StringRef getTypeAttrName() { return "nodetype"; }
      TypeAttr getTypeAttr() {
	return getOperation()->getAttrOfType<TypeAttr>(getTypeAttrName());
      }
      /// Return the one region which may contain code
      Region& getActiveRegion();
      
    };
    
  }
}

#endif
