// -*- C++ -*- //

#ifndef SYNC_NODE_H
#define SYNC_NODE_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
// Needed to include the RegionKindInterface and its trait
#include "mlir/IR/RegionKindInterface.h" 
#include "NodeType.h"
#include "SignalTypes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include <assert.h>

namespace mlir {
  namespace sync {

    class NodeOp : public Op <
      NodeOp,
      SymbolOpInterface::Trait,
      OpTrait::ZeroOperands,
      OpTrait::ZeroResult,
      OpTrait::OneRegion,
      OpTrait::IsIsolatedFromAbove,
      OpTrait::ZeroSuccessor,
      CallableOpInterface::Trait > {
      
    public:      

      using Op::Op;

      static StringRef getOperationName() { return "sync.node"; }
      static void build(OpBuilder &, OperationState &, StringRef,
      			ArrayRef<Type>, ArrayRef<Type>, ArrayRef<Type>,
			bool, ArrayRef<Type>);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      LogicalResult verify();

      /// Type Handling
      
      NodeType getType() { return getTypeAttr().getValue().cast<NodeType>(); }

      /// Regions handling

      Region *getCallableRegion();
      /// Get the body if exists ; if not, fails
      Region& getBody();
      /// Check if the body exists
      bool hasBody() { return !getRegion().empty(); }
      /// Get the operation containing the main loop of the node
      Operation* getMainLoop();
      /// Get the main block (inside the main loop)
      Block* getMainBlock();

      /// Interface handling ; static, inputs and outputs fields

      StringRef getNodeName() ;

      unsigned getNumStatic() { return getType().getNumStatic() ; }
      unsigned getNumInputs() { return getType().getNumInputs() ; }
      unsigned getNumOutputs() { return getType().getNumOutputs() ; }

      ArrayRef<Type> getStaticTypes() { return getType().getStatics(); }
      ArrayRef<Type> getInputsTypes() { return getType().getInputs(); }
      ArrayRef<Type> getOutputsTypes() { return getType().getOutputs(); }
      ArrayRef<Type> getCallableResults() { return getOutputsTypes(); }
      
      iterator_range<Block::args_iterator> getStatics();
      iterator_range<Block::args_iterator> getInputs();
      iterator_range<Block::args_iterator> getOutputs();
      Value getOutput(unsigned i);

    private:
      
      /// Type handling
      
      static StringRef getTypeAttrName() { return "nodetype"; }
      TypeAttr getTypeAttr() {
      	return getOperation()->getAttrOfType<TypeAttr>(getTypeAttrName());
      }

      /// Interface handling
      
      iterator_range<Block::args_iterator> getArguments();
      Value getArgument(unsigned i);
    };
    
  }
}

#endif
