#include "SelectOp.h"

namespace mlir {
  namespace sync {

    void SelectOp::build(OpBuilder &builder, OperationState &result,
			 Value cond, Value trueInput, Value falseInput) {
      result.addOperands({cond, trueInput, falseInput});
      result.addTypes({trueInput.getType()});
    }

    ParseResult SelectOp::parse(OpAsmParser &parser, OperationState &result) {
      auto &builder = parser.getBuilder();
      OpAsmParser::OperandType cond;
      OpAsmParser::OperandType br1;
      OpAsmParser::OperandType br2;
      Type t;
      if (parser.parseOptionalAttrDict(result.attributes) ||
	  parser.parseOperand(cond) ||
	  parser.resolveOperand(cond, builder.getI1Type(), result.operands) ||
	  parser.parseOperand(br1) ||
	  parser.parseOperand(br2) ||
	  parser.parseColonType(t) ||
	  parser.resolveOperand(br1, t, result.operands) ||
	  parser.resolveOperand(br2, t, result.operands))
	return failure();
      result.addTypes({t});
      return failure();
    }

    LogicalResult SelectOp::verify() {
      OpBuilder builder(getOperation());
      if (getCondition().getType() != builder.getI1Type()) {
	return emitOpError() << "Condition must be boolean";
      }
      if (getTrueBranch().getType() != getFalseBranch().getType()) {
	return emitOpError() << "True branch and false branch must have the "
			     << "same type";
      }
      if (getTrueBranch().getType() != getResult().getType()) {
	return emitOpError() << "Inputs and output must have the "
			     << "same type";
      }
      return success();
    }

    void SelectOp::print(OpAsmPrinter &p) {
      p << getOperationName() << " ";
      p.printOptionalAttrDict(getOperation()->getAttrs());
      p << " " << getCondition()
	<< " " << getTrueBranch()
	<< " " << getFalseBranch()
	<< " : " << getResult().getType();
    }
  }
}
