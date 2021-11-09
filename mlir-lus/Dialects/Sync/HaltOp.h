// -*- C++ -*- //

#ifndef SYNC_HALT_H
#define SYNC_HALT_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
  namespace sync {

    class HaltOp : public Op <
      HaltOp,
      OpTrait::ZeroSuccessor,
      OpTrait::IsTerminator> {
      
    public:
      
      using Op::Op;

      static StringRef getOperationName() { return "sync.halt" ; }
      static void build(Builder &builder, OperationState &state) {}
      static ParseResult parse(OpAsmParser &parser, OperationState &result) {
	return success();
      }
      void print(OpAsmPrinter &p) { p << getOperationName() ; }
      LogicalResult verify() { return success(); }
      static StringRef getFunctionName() { return "halt"; }
    };
  }
}


#endif
