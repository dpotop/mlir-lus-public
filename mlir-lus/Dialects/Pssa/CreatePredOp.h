// -*- C++ -*- //

#ifndef CREATE_PRED_OP_H
#define CREATE_PRED_OP_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "../Lus/KPeriodic.h"
#include "CreatePredType.h"
#include "../MinMaxOperands.h"

namespace mlir {
  namespace pssa {

    class CreatePredOp : public Op <
      CreatePredOp,
      OpTrait::NResults<2>::Impl,
      OpTrait::MinMaxOperands<0,1>::Impl,
      OpTrait::ZeroSuccessor> {
    
    public:
      
      using Op::Op;

      static StringRef getOperationName() { return "pssa.create_pred"; }
      static void build(Builder *odsBuilder, OperationState &odsState,
			Value complement);
      static void build(Builder *odsBuilder, OperationState &odsState,
			KPeriodic word);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      LogicalResult verify();

      /// Check if the predicate is data-dependant
      bool isDataDependent() { return getType().isData(); }
      /// Check if the predicate is kperiodic
      bool isKPeriodic() { return getType().isKPeriodic(); }
      /// Get if exists the data-dependant condition
      Value  data();
      /// Get if exists the kperiodic condition
      lus::KPeriodic word();
      /// Get the positive predicate
      Value getTruePred() { return getResult(0); }
      /// Get the negative predicate
      Value getFalsePred() { return getResult(1); }
      /// Assuming that p is one of the predicates, get the other one
      Value getComplementPred(Value p);
      
    private:
      
      static StringRef getTypeAttrName() { return "createpredtype"; }
      TypeAttr getTypeAttr() {
	return getOperation()->getAttrOfType<TypeAttr>(getTypeAttrName());
      }
      CreatePredType getType() {
	return getTypeAttr().getValue().cast<CreatePredType>();
      }
    };
    
  }
}

#endif
