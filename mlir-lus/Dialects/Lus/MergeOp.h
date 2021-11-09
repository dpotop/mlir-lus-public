// -*- C++ -*- //

#ifndef MERGEOP_H
#define MERGEOP_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "WhenType.h"
#include "../MinMaxOperands.h"

namespace mlir {
  namespace lus {
    
    class MergeOp : public Op <MergeOp,
			       OpTrait::OneResult,
			       OpTrait::MinMaxOperands<2,3>::Impl,
			       OpTrait::ZeroSuccessor > {
    public:

      using Op::Op;

      static StringRef getOperationName() { return "lus.merge"; }
      /// The type of the input data
      Type getDataType() { return getType().getDataType() ; }
      /// The type of the condition (kperiodic word or data-dependant)
      Cond<Type> getCondType() { return getType().getCondType() ; }
      /// The value on true stream
      Value getTrueInput() { return getOperand(0) ; }
      /// The value on false stream
      Value getFalseInput() { return getOperand(1) ; }
      /// Check if the condition is data-dependant
      bool isCondData() { return getCondType().getType() == CondDataType; }
      /// Check if the condition is kperiodic
      bool isCondKPeriodic() { return getCondType().getType() == CondKPType; }
      /// Get the condition if it is data-dependant
      Value getCondValue();
      /// Get the condition if it is kperiodic
      KPeriodic getCondKPeriodic();

      static void build(OpBuilder &builder, OperationState &result,
			Cond<Value> cond, Value trueInput, Value falseInput);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      LogicalResult verify() ;
      void print(OpAsmPrinter &p);

    private:

      static StringRef getTypeAttrName() { return "whentype"; }
      TypeAttr getTypeAttr() {
      	return getOperation()->getAttrOfType<TypeAttr>(getTypeAttrName());
      }
      WhenType getType() {
	return getTypeAttr().getValue().template cast<WhenType>();
      }
    };
  }
}

#endif
