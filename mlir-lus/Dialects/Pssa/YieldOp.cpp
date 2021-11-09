#include "YieldOp.h"
#include "CondactOp.h"

namespace mlir {
  namespace pssa {

    void YieldOp::build(Builder &odsBuilder,
			OperationState &odsState,
			ValueRange results) {
      odsState.addOperands(results);
    }
    
    ParseResult YieldOp::parse(OpAsmParser &parser, OperationState &result) {
      SmallVector<OpAsmParser::OperandType, 4> operands;
      SmallVector<Type, 4> types;
      llvm::SMLoc loc = parser.getCurrentLocation();
      // Parse variadic operands list, their types, and resolve operands to
      // SSA values.
      if (parser.parseOptionalAttrDict(result.attributes) ||
	  parser.parseOperandList(operands) ||
	  parser.parseOptionalColonTypeList(types) ||
	  parser.resolveOperands(operands, types, loc, result.operands))
	return failure();
      return success();
    }

    void YieldOp::print(OpAsmPrinter &p) {
      p << getOperationName();
      p.printOptionalAttrDict(getOperation()->getAttrs());
      if (getNumOperands() != 0)
	p << ' ' << getOperands() << " : " << getOperandTypes();
    }

    LogicalResult YieldOp::verify() {

      auto parentOp = getOperation()->getParentOp();
      auto results = parentOp->getResults();
      auto operands = getOperands();

      if (isa<CondactOp>(parentOp)) {
	if (parentOp->getNumResults() != getNumOperands())
	  return emitOpError() << "parent of yield must have same number of "
	    "results as the yield operands";
	for (auto e : llvm::zip(results, operands)) {
	  if (std::get<0>(e).getType() != std::get<1>(e).getType())
	    return emitOpError()
	      << "types mismatch between yield op and its parent";
	}
      } else {
	return emitOpError()
	  << "yield only terminates Condact regions";
      }

      return success();
    }
  }
}
