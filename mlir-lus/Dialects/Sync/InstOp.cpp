#include "InstOp.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
  namespace sync {

    LogicalResult InstOp::verify() {

      Operation *calleeNodePtr = getCalleeNode();
      if (!isa<NodeOp>(calleeNodePtr))
      	return emitOpError() << getCalleeName() << " must be a node.";
      NodeOp calleeNode = dyn_cast<NodeOp>(calleeNodePtr);
      NodeType calleeType = calleeNode.getType();
      auto calleeInputTypes = calleeType.getInputs();
      auto calleeOutputTypes = calleeType.getOutputs();

      if (llvm::size(calleeInputTypes) != llvm::size(getOperands())) {
      	return
      	  emitOpError()
      	  << getCalleeName() << " should have "
	  << llvm::size(calleeInputTypes)
      	  << " operands but is istancied with "
      	  << llvm::size(getOperands());
      }
      if (llvm::size(calleeOutputTypes) != llvm::size(getResults())) {
      	return
      	  emitOpError()
      	  << getCalleeName() << " should have "
	  << llvm::size(calleeOutputTypes)
      	  << " results but is istancied with "
      	  << llvm::size(getResults());
      }
      int i = 0;
      for (auto e : llvm::zip(calleeInputTypes, getOperands().getTypes())) {
	Type t = std::get<0>(e).cast<SiginType>().getType();
      	if (t != std::get<1>(e)) {
      	  return
      	    emitOpError()
      	    << "operand n#" << i << " should have type "
      	    << t << " but has " << std::get<1>(e);
      	}
      }
      i = 0;
      for (auto e : llvm::zip(calleeOutputTypes, getResults().getTypes())) {
	Type t = std::get<0>(e).cast<SigoutType>().getType();
      	if (t != std::get<1>(e)) {
      	  return
      	    emitOpError()
      	    << "result n#" << i << " should have type "
      	    << t << " but has " << std::get<1>(e);
      	}
      }
      
      if (!getOperation()->getAttr(getIdAttrName()) ||
	  !getOperation()->getAttr(getIdAttrName()).cast<IntegerAttr>())
	return emitOpError() << "Instance id must be set and integer";
      return success();
    }

    void InstOp::build(OpBuilder &builder, OperationState &state,
		       StringRef name, int64_t id,
		       ArrayRef<Value> params, ArrayRef<Type> results) {
      auto context = builder.getContext();
      FlatSymbolRefAttr calleeAttr = FlatSymbolRefAttr::get(context, name);
      state.addAttribute("callee", calleeAttr);
      IntegerAttr idAttr = IntegerAttr::get(builder.getI32Type(), id);
      state.addAttribute(getIdAttrName(), idAttr);
      state.addOperands(params);
      state.addTypes(results);
    }
    
    ParseResult InstOp::parse(OpAsmParser &parser,
			      OperationState &result) {
      SmallVector<OpAsmParser::OperandType, 4> inputs;
      FunctionType ft;
      llvm::SMLoc loc = parser.getCurrentLocation();
      FlatSymbolRefAttr calleeAttr;
      IntegerAttr idAttr;

      if (parser.parseAttribute(calleeAttr, "callee", result.attributes) ||
	  parser.parseAttribute(idAttr, getIdAttrName(), result.attributes) ||
	  parser.parseLParen() ||
	  parser.parseOperandList(inputs) ||
	  parser.parseRParen() ||
	  parser.parseColonType(ft))
	return failure();

      if (parser.resolveOperands(inputs, ft.getInputs(),
				 loc, result.operands))
	return failure();
      
      result.addTypes(ft.getResults());

      return success();
    }

    void InstOp::print(OpAsmPrinter &p) {
      p << getOperationName() << " ";
      p.printAttribute(getCallee());
      p << " " << getId();
      p << " (";
      {
	unsigned i = 0;
	for (Value v: getOperands()) {
	  if (i > 0)
	    p << ", ";
	  p << v;
	  i++;
	}
      }
      p << ") : (" << getOperands().getTypes() << ")"
	<< " -> " <<  getResults().getTypes();
    }

    Operation* InstOp::getCalleeNode() {
      Operation* rootOp = getOperation();
      while (!isa<ModuleOp>(rootOp)) {
	rootOp = rootOp->getParentOp();
      }
      Operation* calleeOp = SymbolTable::lookupSymbolIn(rootOp,
      							getCalleeName());
      assert(calleeOp);
      return calleeOp;
    }

    int64_t InstOp::getId() {
      return
	getOperation()
	->getAttr(getIdAttrName()).cast<IntegerAttr>().getInt();
    }
    
  }
}
