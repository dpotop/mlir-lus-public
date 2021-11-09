// -*- C++ -*- //

#ifndef OUTPUT_OP_H
#define OUTPUT_OP_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "Sync.h"

namespace mlir {
  namespace sync {

    class OutputOp: public Op <
      OutputOp,
      OpTrait::NOperands<2>::Impl,
      OpTrait::OneResult,
      OpTrait::ZeroSuccessor > {
    private:

    public:

      using Op::Op;

      static StringRef getOperationName() { return "sync.output"; }
      static void build(Builder &odsBuilder,
			OperationState &odsState,
			Value sig, Value v);
      void print(OpAsmPrinter &p);
      LogicalResult verify();
      static ParseResult parse(OpAsmParser &parser, OperationState &result);

      /// Get the output signal
      Value getSignal() { return getOperand(0); }
      /// Get the parameter sent to the output signal
      Value getParameter() { return getOperand(1); }
      
      
    };
    
  }
}

#endif
