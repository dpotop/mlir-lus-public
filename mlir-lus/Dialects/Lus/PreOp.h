// PreOp class definitions -*- C++ -*- //

#ifndef MLIRLUS_PRE_H
#define MLIRLUS_PRE_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
  namespace lus {

    class PreOp : public Op <
      PreOp,
      OpTrait::OneResult,
      OpTrait::ZeroSuccessor,
      OpTrait::SameOperandsAndResultType,
      OpTrait::OneOperand> {
    public:
      using Op::Op;

      static StringRef getOperationName() { return "lus.pre"; }
      static void build(Builder &builder, OperationState &state, Value v);
      LogicalResult verify();
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
    };
  }
}

#endif
