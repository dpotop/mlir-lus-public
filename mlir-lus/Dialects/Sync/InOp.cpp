#include "InOp.h"
#include "SignalTypes.h"

namespace mlir {
  namespace sync {

    void InOp::build(Builder &odsBuilder, OperationState &odsState,
		     Value arg) {
      odsState.addOperands(arg);
      odsState.addTypes(SiginType::get(odsBuilder.getContext(),
				       arg.getType()));
    }

    ParseResult InOp::parse(OpAsmParser &parser, OperationState &result) {
      auto builder = parser.getBuilder();
      OpAsmParser::OperandType op;
      Type t;
      if (parser.parseOperand(op)
	  || parser.parseColonType(t)
	  || parser.resolveOperand(op, t, result.operands))
	return failure();
      parser.addTypeToList(SiginType::get(builder.getContext(), t),
			   result.types);
      return success();
    }

    void InOp::print(OpAsmPrinter &p) {
      p << getOperationName() << " "
	<< getOperand()
	<< " : " << getOperand().getType();
    }

    LogicalResult InOp::verify() {
      Type rt = getResult().getType();
      if (!rt.isa<SiginType>())
	return emitOpError() << getOperationName()
			     << "must return a sigin type";

      SiginType st = rt.cast<SiginType>();
      if (getOperand().getType() != st.getType())
	return emitOpError() << "operand must be " << st.getType();
	
      return success();
    }
  }
}
