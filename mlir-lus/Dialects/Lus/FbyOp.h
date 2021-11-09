// FbyOp class definitions -*- C++ -*- //

#ifndef MLIRLUS_FBY_H
#define MLIRLUS_FBY_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
  namespace lus {

    class FbyOp : public Op <
      FbyOp,
      OpTrait::OneResult,
      OpTrait::ZeroSuccessor,
      OpTrait::SameOperandsAndResultShape,
      OpTrait::NOperands<2>::Impl> {
    public:
      using Op::Op;

      static StringRef getOperationName() { return "lus.fby"; }

      Value getLhs() { return getOperand(0) ; }
      Value getRhs() { return getOperand(1) ; }

      LogicalResult verify();
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
    };
  }
}

#endif
