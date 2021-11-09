#include "TickOp.h"

namespace mlir {
  namespace sync {

    void TickOp::build(Builder &odsBuilder, OperationState &odsState,
		       ArrayRef<Value> vs) {
      odsState.addOperands(vs);
      odsState.addTypes(odsBuilder.getI32Type());
    }

    ParseResult TickOp::parse(OpAsmParser &parser, OperationState &result) {
      auto builder = parser.getBuilder();
      
      SmallVector<OpAsmParser::OperandType, 4> operands;
      SmallVector<Type, 4> types;
      if (parser.parseLParen() ||
	  parser.parseOperandList(operands) ||
	  parser.parseRParen() ||
	  parser.parseColonTypeList(types))
	return failure();
      
      for (auto e: llvm::zip(operands, types)) {
	parser.resolveOperand(std::get<0>(e), std::get<1>(e),
			      result.operands);
      }
      result.addTypes(builder.getI32Type());
      
      return success();
    }

    void TickOp::print(OpAsmPrinter &p) {
      p << getOperationName() << "(";
      p.printOperands(getOperands());
      p << ") : " << getResult().getType();
    }

    LogicalResult TickOp::verify() {
      return success();
    }
    
  }
}
