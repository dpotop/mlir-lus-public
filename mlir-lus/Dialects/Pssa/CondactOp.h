// -*- C++ -*- //

#ifndef CONDACT_OP_H
#define CONDACT_OP_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "YieldOp.h"

namespace mlir {
  namespace pssa {

    class CondactOp : public Op <
      CondactOp,
      OpTrait::VariadicResults,
      OpTrait::ZeroSuccessor,
      OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl,
      OpTrait::AtLeastNOperands<1>::Impl > {
    public:
      using Op::Op;

      static StringRef getOperationName() { return "pssa.condact"; }
      static void build(Builder& builder, OperationState &result,
			Value cond, ArrayRef<Type> resultTypes);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      LogicalResult verify();
      
      /// Get the activation condition
      Value  condition() { return getOperand(0); }
      /// Check if default values have been set
      bool hasDefaults() { return getNumDefaults(); }
      /// Get the number of default values
      unsigned getNumDefaults() { return getOperands().size() - 1; }
      /// Get the default values
      Operation::operand_range defaults(){return getOperands().drop_front(1);}
      // Get the body of the condact
      Region &getBody() { return getOperation()->getRegion(0); }
      // Get the final yield of the condact
      YieldOp getYield();
      
    };

  }
}

#endif
