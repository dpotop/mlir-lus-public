#include "CondactOp.h"

namespace mlir {
  namespace pssa {

    YieldOp CondactOp::getYield() {
      Operation* op = getBody().back().getTerminator();
      assert(isa<YieldOp>(op));
      YieldOp yieldOp = dyn_cast<YieldOp>(op);
      return yieldOp;
    }
    
    ParseResult CondactOp::parse(OpAsmParser &parser,
				 OperationState &result) {
      
      auto &builder = parser.getBuilder();
      
      Region *body = result.addRegion();
      OpAsmParser::OperandType cond;
      SmallVector<OpAsmParser::OperandType, 4> defaults;
      
      if (// Parse the condition
	  parser.parseOperand(cond) ||
	  parser.resolveOperand(cond, builder.getI1Type(), result.operands) ||
	  // Parse the default values
	  parser.parseOptionalKeyword("default") ||
	  parser.parseOptionalLParen() ||
	  parser.parseOperandList(defaults) ||
	  parser.parseOptionalRParen() ||
	  // Parse the condact region
	  parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}) ||
	  parser.parseOptionalAttrDict(result.attributes) ||
	  // Parse the output types
	  parser.parseColonTypeList(result.types))
	return failure();
      
      for (auto operand_type : llvm::zip(defaults, result.types)) {
	if (parser.resolveOperand(std::get<0>(operand_type),
				  std::get<1>(operand_type),
				  result.operands)) {
	  return failure();
	}
      }

      CondactOp::ensureTerminator(*body, builder, result.location);
      
      return success();
    }

    void CondactOp::print(OpAsmPrinter &p) {

      p << getOperationName();
      p.printOptionalAttrDict(getOperation()->getAttrs());
      p << " " << condition() ;
      if (getNumDefaults() > 0)
	p << " default(" << defaults() << ")";
      
      p.printRegion(getBody(),
		    /*printEntryBlockArgs=*/false,
		    /*printBlockTerminators=*/true);
      p.printOptionalAttrDict(getOperation()->getAttrs());
      p << " : (" << getResultTypes() << ")";
    }

    LogicalResult CondactOp::verify() {
      const unsigned numDefaults = getNumDefaults();
      const unsigned numResults = getResultTypes().size();
      if (numDefaults != numResults && numDefaults > 0) {
      	return
      	  emitOpError() << "condact has "
      			<< numDefaults << " default variables "
      			<< " but should have " << numResults;
      }
      else if (numDefaults > 0) {
	for (auto e : llvm::zip(defaults(), getResultTypes())) {
	  if (std::get<0>(e).getType() != std::get<1>(e))
	    return emitOpError()
	      << "types mismatch between default values and result types";
	}
      }
      return success();
    }

    void CondactOp::build(Builder& builder, OperationState &result,
			  Value cond, ArrayRef<Type> resultTypes) {
      result.addOperands(cond);
      result.addTypes(resultTypes);
      Region *bodyRegion = result.addRegion();
      CondactOp::ensureTerminator(*bodyRegion, builder, result.location);
    }
  }
}
