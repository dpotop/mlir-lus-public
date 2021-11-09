#include "OutOp.h"
#include "SignalTypes.h"

namespace mlir {
  namespace sync {

    void OutOp::build(Builder &odsBuilder, OperationState &odsState,
		      Value arg) {
      odsState.addOperands(arg);
      odsState.addTypes(SigoutType::get(odsBuilder.getContext(),
					arg.getType()));
    }

    ParseResult OutOp::parse(OpAsmParser &parser, OperationState &result) {
      auto builder = parser.getBuilder();
      OpAsmParser::OperandType op;
      Type t;
      if (parser.parseOperand(op)
	  || parser.parseColonType(t)
	  || parser.resolveOperand(op, t, result.operands))
	return failure();
      parser.addTypeToList(SigoutType::get(builder.getContext(), t),
			   result.types);
      return success();
    }

    void OutOp::print(OpAsmPrinter &p) {
      p << getOperationName() << " "
	<< getOperand()
	<< " : " << getOperand().getType();
    }

    LogicalResult OutOp::verify() {
      if (!getResult().getType().isa<SigoutType>()) {
	return emitOpError() << getOperationName()
			     << " must return a signal";
      }
      SigoutType st = getResult().getType().cast<SigoutType>();
      if (st.getType() != getOperand().getType()) {
	return emitOpError() << "Operand must be " << st.getType()
			     << " but is " << getOperand().getType();
      }
      return success();
    }
  }
}
