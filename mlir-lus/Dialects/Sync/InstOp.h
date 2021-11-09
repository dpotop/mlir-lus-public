// -*- C++ -*- //

#ifndef INST_OP_H
#define INST_OP_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
// #include "mlir/IR/Function.h"
#include "Node.h"
#include "../../Tools/ParserAux.h"

namespace mlir {
  namespace sync {

    class InstOp: public Op <
      InstOp,
      OpTrait::VariadicResults,
      OpTrait::VariadicOperands,
      CallOpInterface::Trait,
      OpTrait::ZeroSuccessor > {

    public:

      using Op::Op;

      static StringRef getOperationName() { return "sync.inst"; }
      LogicalResult verify();
      void static build(OpBuilder &builder, OperationState &state,
			StringRef calleeName, int64_t id,
			ArrayRef<Value> params, ArrayRef<Type> results);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);

      /// Required by the interface
      CallInterfaceCallable getCallableForCallee() { return getCallee() ;}
      /// Get the callee (generic when lowering replaces node by func)
      Operation* getCalleeNode();
      /// Get outputs and inputs values
      operand_range getArgOperands() {return {operand_begin(),operand_end()};}
      /// Get the instance id
      int64_t getId();

      StringRef getCalleeName() { return getCallee().getValue() ; }
      
    private:

      static StringRef getIdAttrName() { return "id"; }
      
      /// Callee management
      
      FlatSymbolRefAttr getCallee() {
	return getOperation()->getAttrOfType<FlatSymbolRefAttr>("callee");
      }
    };
  }
}

#endif
