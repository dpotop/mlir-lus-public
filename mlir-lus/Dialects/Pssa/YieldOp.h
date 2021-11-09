// -*- C++ -*- //

#ifndef YIELD_OP_H
#define YIELD_OP_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
  namespace pssa {

    class YieldOp : public Op <
      YieldOp,
      OpTrait::ZeroResult,
      OpTrait::ZeroSuccessor,
      OpTrait::IsTerminator,
      OpTrait::VariadicOperands > {
    public:
      using Op::Op;

      static StringRef getOperationName() { return "pssa.yield"; }
      static void build(Builder &, OperationState &,
			ValueRange results = llvm::None);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      LogicalResult verify();
    };
  }
}

#endif
