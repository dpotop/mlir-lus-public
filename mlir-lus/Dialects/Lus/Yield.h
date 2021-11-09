// -*- C++ -*- //

#ifndef MLIRLUS_YIELD_NEW_H
#define MLIRLUS_YIELD_NEW_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "YieldType.h"



namespace mlir {
  namespace lus {

    class YieldOp : public Op <
      YieldOp,
      OpTrait::ZeroSuccessor,
      OpTrait::IsTerminator,
      OpTrait::VariadicOperands > {
      
    public:
      
      using Op::Op;

      static StringRef getOperationName() { return "lus.yield" ; }
      static void build(Builder &builder, OperationState &state,
			ArrayRef<Value> states, ArrayRef<Value> outputs);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      LogicalResult verify();

      /// Operands management : outputs and state values fields
      
      unsigned getNumStates() { return getType().getNumStates() ; }
      unsigned getNumOutputs() { return getType().getNumOutputs() ; }
      ArrayRef<Type> getStatesTypes() { return getType().getStates(); }
      ArrayRef<Type> getOutputsTypes() { return getType().getOutputs(); }
      Operation::operand_range getStates();
      Operation::operand_range getOutputs();
      /// Add a state value to the yield
      void addState(Value state);

    private:

      static StringRef getTypeAttrName() { return "yieldtype"; }
      TypeAttr getTypeAttr() {
	return getOperation()->getAttrOfType<TypeAttr>(getTypeAttrName());
      }
      YieldType getType() {
	return getTypeAttr().getValue().cast<YieldType>();
      }

    };
  }
}


#endif
