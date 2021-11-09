#include "OutputOp.h"
#include "SignalTypes.h"

namespace mlir {
  namespace sync {

    void OutputOp::build(Builder &odsBuilder,
			 OperationState &odsState,
			 Value sig, Value v) {
      SigoutType st = sig.getType().cast<SigoutType>();
      assert(st.getType() == v.getType());
      odsState.addOperands(sig);
      odsState.addOperands(v);
      odsState.addTypes(odsBuilder.getI32Type());
    }

    ParseResult OutputOp::parse(OpAsmParser &parser, OperationState &result) {
      auto builder = parser.getBuilder();
      Type t;
      OpAsmParser::OperandType op1;
      OpAsmParser::OperandType op2;
      if (parser.parseLParen()
	  || parser.parseOperand(op1)
	  || parser.parseColon()
	  || parser.parseOperand(op2)
	  || parser.parseRParen()
	  || parser.parseColonType(t))
	return failure();
      SigoutType st = builder.getType < SigoutType, Type > (t);
      if (parser.resolveOperand(op1, st, result.operands)
	  || parser.resolveOperand(op2, t, result.operands))
	return failure();
      parser.addTypeToList(builder.getI32Type(), result.types);
      return success();
    }

    void OutputOp::print(OpAsmPrinter &p) {
      p << getOperationName() << " ("
	<< getOperand(0) << " : " << getOperand(1) << ")";
      p << " : " << getOperand(1).getType();
    }

    LogicalResult OutputOp::verify() {
      OpBuilder builder(getOperation());
      if (getResult().getType() != builder.getI32Type()) {
	return emitOpError() << getOperationName()
			     << " must return i32 but returns "
			     << getResult().getType();
      }
      if (!getSignal().getType().isa<SigoutType>()) {
	return emitOpError() << getOperationName()
			     << " needs a signal";
      }
      SigoutType st = getSignal().getType().cast<SigoutType>();
      if (st.getType() != getParameter().getType()) {
	return emitOpError() << "Parameter must be " << st.getType()
			     << " but is " << getParameter().getType();
      }
      return success();
    }
    
  }
}
