#include "PreOp.h"

namespace mlir {
  namespace lus {

    LogicalResult PreOp::verify() {
      return mlir::success() ;
    }

    void PreOp::build(Builder &builder, OperationState &state, Value v) {
      state.addOperands(v);
      state.addTypes(v.getType());
    }
    
    ParseResult PreOp::parse(OpAsmParser &parser, OperationState &result) {
      OpAsmParser::OperandType op;
      Type t;

      if (parser.parseOperand(op)
	  || parser.parseColonType(t))
	return failure();

      if (parser.resolveOperands({op}, {t}, parser.getCurrentLocation(),
				 result.operands))
	return failure();
      result.addTypes(t);

      if (parser.parseOptionalAttrDict(result.attributes))
	return failure();
	      
      return success();
    }
    
    void PreOp::print(OpAsmPrinter &p) {
      Value i = getOperand() ;
      p << "lus.pre " << i << " : " << i.getType() ;
      p.printOptionalAttrDict(getOperation()->getAttrs(), /*elidedAttrs=*/{});
    }

  }
}
