// -*- C++ -*- //

#ifndef INPUT_OP_H
#define INPUT_OP_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

namespace mlir {
  namespace sync {

    class InputOp: public Op <
      InputOp,
      OpTrait::OneResult,
      OpTrait::OneOperand,
      OpTrait::ZeroSuccessor > {

    public:

      using Op::Op;

      static StringRef getOperationName() { return "sync.input"; }
      static void build(Builder &, OperationState &, Value);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      LogicalResult verify();

      /// Get the input signal
      Value getSignal() { return getOperand(); }
    };
    
  }
}

#endif
