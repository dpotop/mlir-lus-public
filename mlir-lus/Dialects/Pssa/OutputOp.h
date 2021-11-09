// -*- C++ -*- //

#include "mlir/Dialect/StandardOps/IR/Ops.h"

#ifndef PSSA_OUTPUT_OP_H
#define PSSA_OUTPUT_OP_H

namespace mlir {
  namespace pssa {

    class OutputOp: public Op < OutputOp,
				OpTrait::OneResult,
				OpTrait::OneOperand,
				OpTrait::ZeroSuccessor > {

      public:

      using Op::Op;

      static StringRef getOperationName() { return "pssa.output"; }

      int64_t getPosition();

      static void build(OpBuilder &builder, OperationState &result,
			int64_t pos, Value v);

      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      
      LogicalResult verify() ;
      
      void print(OpAsmPrinter &p);

    private:

      static StringRef getPosKey() { return "pos"; }
    };
    
  }
}

#endif
