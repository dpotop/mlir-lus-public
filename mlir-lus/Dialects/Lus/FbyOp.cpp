#include "FbyOp.h"

namespace mlir {
  namespace lus {

    LogicalResult FbyOp::verify() {
      Value o = getResult() ;
      Value lhs = getLhs() ;
      Value rhs = getRhs() ;

      if (rhs.getType() != lhs.getType()) {
	return emitOpError() << "Operand #1 and operand #2 "
			     << "must share the same type." ;
      }
      if (rhs.getType() != o.getType()) {
	return emitOpError() << "Operand #1 and result "
			     << "must share the same type." ;
      }
      return mlir::success() ;
    }

    void FbyOp::build(Builder &builder, OperationState &state, Value v1, Value v2) {
      state.addOperands(v1);
      state.addOperands(v2);
      state.addTypes(v1.getType());
    }

    ParseResult FbyOp::parse(OpAsmParser &parser, OperationState &result) {
      OpAsmParser::OperandType inputOperands[2];
      Type t;

      if (parser.parseOperand(inputOperands[0])
	  || parser.parseOperand(inputOperands[1])
	  || parser.parseColonType(t))
	return failure();

      if (parser.resolveOperands(inputOperands, {t, t},
				 parser.getCurrentLocation(),
				 result.operands))
	return failure();

      if (parser.parseOptionalAttrDict(result.attributes))
	return failure();
      
      result.addTypes(t);
	      
      return success();
    }
    
    void FbyOp::print(OpAsmPrinter &p) {
      Value lhs = getLhs() ;
      Value rhs = getRhs() ;
      p << "lus.fby";
      p << " " << lhs << " " << rhs << ":" << getResult().getType();
      p.printOptionalAttrDict(getOperation()->getAttrs(), /*elidedAttrs=*/{});
      
    }
  }
}
