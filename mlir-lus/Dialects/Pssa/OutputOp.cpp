#include "OutputOp.h"

namespace mlir {
  namespace pssa {

    int64_t OutputOp::getPosition() {
      Attribute attr = getOperation()->getAttrDictionary().get(getPosKey());
      assert(attr != NULL);
      assert(attr.isa<IntegerAttr>());
      IntegerAttr integerAttr = attr.cast<IntegerAttr>();
      return integerAttr.getValue().getLimitedValue();
    }

    void OutputOp::build(OpBuilder &builder, OperationState &result,
			 int64_t pos, Value v) {
      result.addAttribute(getPosKey(),
			  builder.getI64IntegerAttr(pos));
      result.addOperands(v);
      result.addTypes(builder.getI32Type());
    }

    ParseResult OutputOp::parse(OpAsmParser &parser, OperationState &result) {
      OpAsmParser::OperandType op;
      Type opType;
      if (parser.parseOptionalAttrDict(result.attributes)
	  || parser.parseLParen()
	  || parser.parseOperand(op)
	  || parser.parseRParen()
	  || parser.parseColon()
	  || parser.parseType(opType)
	  || parser.resolveOperand(op, opType, result.operands))
	return failure();
      result.addTypes({parser.getBuilder().getI32Type()});
      return success();
    }

    LogicalResult OutputOp::verify() {
      if (getOperation()->getAttrDictionary().get(getPosKey()) == NULL) {
	return emitOpError() << getPosKey() << "attribute needed.";
      }
      return success();
    }

    void OutputOp::print(OpAsmPrinter &p) {
      p << getOperationName() << " " << getOperation()->getAttrDictionary()
	<< "(" << getOperand() << "): "
	<< getOperand().getType();
    }
  }
}
