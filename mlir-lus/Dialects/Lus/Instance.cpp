#include "Instance.h"
#include "Node.h"

namespace mlir {
  namespace lus {

    bool InstanceOp::mustInline() {
      return
	getOperation()
	->getAttr(getInlineAttrName()).cast<BoolAttr>()
	.getValue();
    }
    
    FlatSymbolRefAttr InstanceOp::getCallee() {
      return getOperation()->getAttrOfType<FlatSymbolRefAttr>("callee");
    }

    NodeOp InstanceOp::getCalleeNode() {
      // Search the callee name at top level
      Operation* rootOp = getOperation();
      while (auto* parentOp = rootOp->getParentOp()) {
	rootOp = parentOp;
      }
      Operation* calleeOp = SymbolTable::lookupSymbolIn(rootOp,
							getCalleeName());
      NodeOp calleeNode = dyn_cast<NodeOp>(calleeOp);
      return calleeNode;
    }
    
    ParseResult InstanceOp::parse(OpAsmParser &parser,
				  OperationState &result) {
      SmallVector<OpAsmParser::OperandType, 4> input;
      FunctionType type;
      ArrayRef<Type> operandsTypes;
      ArrayRef<Type> allResultTypes;
      FlatSymbolRefAttr calleeAttr;
      llvm::SMLoc loc = parser.getCurrentLocation();
      

      bool mustInline = succeeded(parser.parseOptionalKeyword("inline"));
      BoolAttr inlineAttr = BoolAttr::get(parser.getBuilder().getContext(),
					  mustInline);
      result.addAttribute(getInlineAttrName(), inlineAttr);
      if (parser.parseAttribute(calleeAttr, "callee", result.attributes))
	return failure() ;
      // state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
      // Operand list 
      if(parser.parseLParen()) return failure() ;
      if(parser.parseOperandList(input)) return failure() ;
      if(parser.parseRParen()) return failure() ;

      // Function type
      if (parser.parseColon()) return failure() ;
      if (parser.parseType(type)) return failure() ;
      operandsTypes = type.getInputs();
      allResultTypes = type.getResults();
      result.addTypes(allResultTypes);

      // Add types to names to create values
      if (parser.resolveOperands(input, operandsTypes,
				 loc, result.operands))
	return failure();
      
      return success();
    }

    LogicalResult InstanceOp::verify() {
      // TODO test if callee exists
      NodeOp calleeNode = getCalleeNode();
      NodeType calleeType = calleeNode.getType();
      auto calleeInputTypes = calleeType.getInputs();
      auto calleeOutputTypes = calleeType.getResults();
      auto instOperandsTypes = getArgOperands().getTypes();
      auto instResultsTypes = getResults().getTypes();
      const auto numCalleeInputTypes = llvm::size(calleeInputTypes);
      const auto numInstOperandsTypes = llvm::size(instOperandsTypes);
      const auto numCalleeOutputTypes = llvm::size(calleeOutputTypes);
      const auto numInstResultsTypes = llvm::size(instResultsTypes);
      if (numCalleeInputTypes != numInstOperandsTypes) {
      	return
      	  emitOpError() << getCalleeName()
      			<< " waits " << numCalleeInputTypes << " parameters "
      			<< "but is instancied with " << numInstOperandsTypes;
      }
      if (numCalleeOutputTypes != numInstResultsTypes) {
      	return
      	  emitOpError() << getCalleeName()
      			<< " produces " << numCalleeOutputTypes << " results "
      			<< "but is istancied with " << numInstResultsTypes;
      }
      int i = 0;
      for (auto e : llvm::zip(calleeInputTypes, instOperandsTypes)) {
      	if (std::get<0>(e) != std::get<1>(e)) {
      	  return
      	    emitOpError() << "operand #" << i << " must have type "
      			  << std::get<0>(e) << " but is " << std::get<1>(e);
      	}
      }
      i = 0;
      for (auto e : llvm::zip(calleeOutputTypes, instResultsTypes)) {
      	if (std::get<0>(e) != std::get<1>(e)) {
      	  return
      	    emitOpError() << "result #" << i << " must have type "
      			  << std::get<0>(e) << " but is " << std::get<1>(e);
      	}
      }
      return success();
    }
    
    void InstanceOp::print(OpAsmPrinter &p) {
      p << "lus.instance";
      p << " ";
      p.printAttribute(getCallee());
      p << " (" << getOperands() << ") : ";
      p.printFunctionalType(getOperation()->getOperandTypes(),
			    getOperation()->getResultTypes());
    }    
  }
}
