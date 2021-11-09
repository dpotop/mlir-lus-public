// -*- C++ -*- //

#ifndef OUT_OP_H
#define OUT_OP_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

namespace mlir {
  namespace sync {

    class OutOp: public Op <
      OutOp,
      OpTrait::OneResult,
      OpTrait::OneOperand,
      OpTrait::ZeroSuccessor > {

    public:
      
      using Op::Op;

      static StringRef getOperationName() { return "sync.out"; }
      static void build(Builder &odsBuilder, OperationState &odsState,
			Value arg);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      LogicalResult verify();
    };
  }
}

#endif
