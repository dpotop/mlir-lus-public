// -*- C++ -*- //

#ifndef WHENOP_H
#define WHENOP_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "WhenType.h"
#include "../MinMaxOperands.h"

namespace mlir {
  namespace lus {
    
    class WhenOp : public Op <WhenOp,
			      OpTrait::OneResult,
			      OpTrait::MinMaxOperands<1,2>::Impl,
			      OpTrait::ZeroSuccessor > {
    public:
      
      using Op::Op;

      static StringRef getOperationName() { return "lus.when"; }

      /// The type of the input data
      Type getDataType() { return getType().getDataType() ; }
      /// The input data
      Value getDataInput() { return getOperand(0) ; }
      /// The type of the condition (kperiodic word or data-dependant)
      Cond<Type> getCondType() { return getType().getCondType() ; }
      /// Check if the condition is data-dependant
      bool isCondData() { return getCondType().getType() == CondDataType; }
      /// Check if the condition is kperiodic
      bool isCondKPeriodic() { return getCondType().getType() == CondKPType; }
      /// Get the condition if it is data-dependant
      Value getCondValue();
      /// Get the condition if it is kperiodic
      KPeriodic getCondKPeriodic();

      LogicalResult verify() ;
      static void build(OpBuilder &builder, OperationState &result,
			Cond<Value> cond, Value input);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      
    private:
      
      static StringRef getTypeAttrName() { return "whentype"; }
      TypeAttr getTypeAttr() {
	return getOperation()->getAttrOfType<TypeAttr>(getTypeAttrName());
      }
      WhenType getType() { return getTypeAttr().getValue().cast<WhenType>(); }
    };
  }
}

#endif
