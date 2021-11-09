// -*- C++ -*- //

#ifndef SYNC_OP_H
#define SYNC_OP_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "Sync.h"

namespace mlir {
  namespace sync {

    class SyncOp: public Op <
      SyncOp,
      OpTrait::VariadicOperands,
      OpTrait::VariadicResults,
      OpTrait::ZeroSuccessor > {

    public:

      using Op::Op;

      static StringRef getOperationName() { return "sync.sync"; }
      static void build(Builder &odsBuilder, OperationState &odsState,
			Value ev, ArrayRef<Value> args);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      LogicalResult verify();

      /// Synchronize on this event
      Value getEvent() { return getOperation()->getOperand(0); }
      /// Copy this value
      OperandRange getValues() {
	return getOperation()->getOperands().drop_front();
      }
      

    };
  }
}

#endif
