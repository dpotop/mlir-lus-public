// -*- C++ -*- //

#ifndef SYNC_SELECT_OP_H
#define SYNC_SELECT_OP_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
  namespace sync {
    
    class SelectOp : public Op <
      SelectOp,
      OpTrait::OneResult,
      OpTrait::NOperands<3>::Impl,
      OpTrait::ZeroSuccessor > {
    public:

      using Op::Op;

      static StringRef getOperationName() { return "sync.select"; }
      Value getCondition() { return getOperand(0); }
      Value getTrueBranch() { return getOperand(1); }
      Value getFalseBranch() { return getOperand(2); }

      static void build(OpBuilder&, OperationState&, Value c,Value t,Value f);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      LogicalResult verify() ;
      void print(OpAsmPrinter &p);
    };
  }
}

#endif
