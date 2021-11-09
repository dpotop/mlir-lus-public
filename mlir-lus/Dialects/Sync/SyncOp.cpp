#include "SyncOp.h"

namespace mlir {
  namespace sync {

    void SyncOp::build(Builder &odsBuilder, OperationState &odsState,
		       Value ev, ArrayRef<Value> args) {
      odsState.addOperands(ev);
      odsState.addOperands(args);
      for (Value arg: args)
	odsState.addTypes(arg.getType());
    }

    ParseResult SyncOp::parse(OpAsmParser &parser, OperationState &result) {
      auto builder = parser.getBuilder();
      OpAsmParser::OperandType ev;
      if (parser.parseLParen()
	  || parser.parseOperand(ev)
	  || parser.resolveOperand(ev,
				   builder.getI32Type(),
				   result.operands))
	return failure();

      SmallVector<Type, 4> ts;
      bool succ = true;
      do {
	OpAsmParser::OperandType a;
	Type at;
	succ = succeeded(parser.parseOptionalComma())
	  && succeeded(parser.parseOperand(a))
	  && succeeded(parser.parseColonType(at));
	if (succ) {
	  if (parser.resolveOperand(a, at, result.operands))
	    return failure();
	  ts.push_back(at);
	}
      } while(succ);

      if (parser.parseRParen())
	return failure();
      for (Type te: ts) {
	parser.addTypeToList(te, result.types);
      }
      return success();
    }

    void SyncOp::print(OpAsmPrinter &p) {
      p << getOperationName()
	<< "(" << getOperation()->getOperand(0);
      for (unsigned i = 1; i < getOperation()->getNumOperands(); i++) {
	p << ", " << getOperation()->getOperand(i)
	  << " : " << getOperation()->getOperand(i).getType();
      }
      p << ")";
    }

    LogicalResult SyncOp::verify() {
      // OpBuilder builder(getOperation());
      // if (getResult().getType() != getValue().getType())
      // 	return emitOpError() << "Operand must be " << getResult().getType()
      // 			     << " but is " << getValue().getType();
      return success();
    }
  }
}
