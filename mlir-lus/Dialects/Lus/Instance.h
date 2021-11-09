// -*- C++ -*- //

#ifndef MLIRLUS_INSTANCE_H
#define MLIRLUS_INSTANCE_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "Node.h"

namespace mlir {
  namespace lus {

    class InstanceOp : public Op <
      InstanceOp,
      OpTrait::VariadicResults,
      OpTrait::ZeroSuccessor,
      CallOpInterface::Trait,
      OpTrait::VariadicOperands > {
    public:
      using Op::Op;

      static StringRef getOperationName() { return "lus.instance"; }
      /// Required by CallOpInterface
      CallInterfaceCallable getCallableForCallee() { return getCallee(); }
      /// The name of the node we intend to instantiate
      StringRef getCalleeName() { return getCallee().getValue() ; }
      /// The node we intend to instantiate
      NodeOp getCalleeNode();

      operand_range getArgOperands() {return {operand_begin(),operand_end()};}
      
      LogicalResult verify();
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      bool mustInline();
    private:
      FlatSymbolRefAttr getCallee();
      static StringRef getInlineAttrName() { return "inline"; }
    };
  }
}

#endif
