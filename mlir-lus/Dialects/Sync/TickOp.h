// -*- C++ -*- //

#ifndef TICK_OP_H
#define TICK_OP_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "../../Dialects/Sync/Sync.h"

namespace mlir {
  namespace sync {

    class TickOp: public Op <
      TickOp,
      OpTrait::OneResult,
      OpTrait::VariadicOperands,
      OpTrait::ZeroSuccessor > {

    public:

      using Op::Op;

      static StringRef getOperationName() { return "sync.tick"; }
      static void build(Builder &odsBuilder,
			OperationState &odsState,
			ArrayRef<Value> vs= {});
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      LogicalResult verify();

      /// When lowering the op, the name of the generated function
      static StringRef getFunctionName() { return "tick"; }
    };

  }
}

#endif
