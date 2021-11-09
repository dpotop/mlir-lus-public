// -*- C++ -*- //

#ifndef IN_OP_H
#define IN_OP_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

namespace mlir {
  namespace sync {

    class InOp: public Op <
      InOp,
      OpTrait::OneResult,
      OpTrait::OneOperand,
      OpTrait::ZeroSuccessor > {

    public:
      
      using Op::Op;

      static StringRef getOperationName() { return "sync.in"; }
      static void build(Builder &odsBuilder, OperationState &odsState,
			Value arg);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      LogicalResult verify();
    };
  }
}

#endif
