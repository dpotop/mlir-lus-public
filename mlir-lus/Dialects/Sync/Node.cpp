#include <algorithm>
// #include "mlir/IR/OperationSupport.h"
#include "../../Tools/CommandLine.h"
#include "../../Tools/ParserAux.h"
#include "../../Transforms/Utilities/OperationsAux.h"
#include "../../Dialects/Sync/UndefOp.h"
#include "Node.h"

using namespace std;

namespace mlir {
  namespace sync {

    /// The sync.node operation

    void NodeOp::build(OpBuilder &builder, OperationState &state,
    		       StringRef name,
    		       ArrayRef<Type> statics,
		       ArrayRef<Type> inputs,
		       ArrayRef<Type> outputs,
		       bool hasBody,
		       ArrayRef<Type> states) {

      state.addAttribute(SymbolTable::getSymbolAttrName(),
    			 builder.getStringAttr(name));
      
      Region *body = state.addRegion();


      SmallVector<Type, 4> inputSig(inputs.size());
      transform (inputs.begin(), inputs.end(), inputSig.begin(),
		 [&, builder] (Type t) {
		   return SiginType::get(builder.getContext(), t);
		 });

      SmallVector<Type, 4> outputSig(outputs.size());
      std::transform (outputs.begin(), outputs.end(), outputSig.begin(),
      		      [&, builder] (Type t) {
      			return SigoutType::get(builder.getContext(), t);
      		      });

      auto type = builder.getType<NodeType,
				  ArrayRef<Type>,
    				  ArrayRef<Type>,
    				  ArrayRef<Type>>(statics,
						  inputSig,
						  outputSig);
      if (hasBody) {
	auto *entry = new Block();
	body->push_back(entry);
	entry->addArguments(type.getStatics());
	entry->addArguments(type.getInputs());
	entry->addArguments(type.getOutputs());
      }
      state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
    }
    
    ParseResult NodeOp::parse(OpAsmParser &parser, OperationState &result) {

      auto builder = parser.getBuilder();

      // Parse the name as a symbol.
      StringAttr nameAttr;
      if (parser.parseSymbolName(nameAttr,
				 ::mlir::SymbolTable::getSymbolAttrName(),
				 result.attributes)) {
	return failure();
      }

      SmallVector<OpAsmParser::OperandType, 4> staticNames;
      SmallVector<Type, 4> staticTys;
      SmallVector<NamedAttrList,4> staticAts; // Currently unused
      SmallVector<OpAsmParser::OperandType, 4> inputNames;
      SmallVector<Type, 4> inputTys;
      SmallVector<NamedAttrList,4> inputAts; // Currently unused
      SmallVector<OpAsmParser::OperandType, 4> outNames;
      SmallVector<Type, 4> outTys;
      SmallVector<NamedAttrList,4> outAts; // Currently unused
      if (succeeded(parser.parseOptionalKeyword("static")))
	if (parseArgumentListParen(parser, staticTys, staticNames, staticAts))
	  return failure();
      if (parseArgumentListParen(parser, inputTys, inputNames, inputAts) ||
	  parser.parseArrow() ||
	  parseArgumentListParen(parser, outTys, outNames, outAts))
	return failure();

      // Parse the optional node region. Start by creating the
      // argument lists (values and types) of the region's first
      // block. This is the concatenation of: staticNames, inputNames, 
      // and the corresponding types.
      SmallVector<OpAsmParser::OperandType, 4> regionArgs;
      regionArgs.append(staticNames.begin(), staticNames.end());
      regionArgs.append(inputNames.begin(), inputNames.end());
      regionArgs.append(outNames.begin(), outNames.end());
      SmallVector<Type, 4> regionArgsTypes;
      regionArgsTypes.append(staticTys.begin(), staticTys.end());
      regionArgsTypes.append(inputTys.begin(), inputTys.end());
      regionArgsTypes.append(outTys.begin(), outTys.end());
      auto *body = result.addRegion();
      if (!parser.parseOptionalRegion(*body, regionArgs, regionArgsTypes).hasValue())
	return failure() ;

      // Build the NodeType of the node, and add it as an attribute to
      // the node.
      auto type = builder.getType<NodeType,
				  ArrayRef<Type>,
				  ArrayRef<Type>,
				  ArrayRef<Type>>(staticTys,
						  inputTys,
						  outTys);

      result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
      
      return success();
    }

    void printAux1NodeOp(OpAsmPrinter &p,
			 ArrayRef<mlir::Type>& stateTypes) {
      p << "(";
      unsigned i = 0 ;
      for(auto t : stateTypes) {
	if (i > 0) p << ", "; i++ ;
	p.printType(t) ;	    
      }
      p << ')';	
    }
    void printAux2NodeOp(OpAsmPrinter &p,
			 iterator_range<Block::args_iterator>& states,
			 ArrayRef<mlir::Type>& stateTypes) {
      p << "(";
      unsigned i = 0 ;
      for(auto pr : llvm::zip(states,stateTypes)) {
	auto val = std::get<0>(pr) ;
	auto typ = std::get<1>(pr) ;
	if (i > 0) p << ", " ; i++ ;
	p.printOperand(val) ;
	p << " : " ;
	p.printType(typ) ;
      }
      p << ')';	
    }
		  
    void NodeOp::print(OpAsmPrinter &p) {

      // print the node name  
      StringRef nodeName = getNodeName() ;
      p << getOperationName() << ' ';

      p.printSymbolName(nodeName);
      
      // Print the signature inputs  
      Region &body = this->getBody();
      bool isExternal = body.empty();

      auto staticTys = getType().getStatics() ;
      if(staticTys.size() != 0) {
	p << " static " ;
	if(isExternal) printAux1NodeOp(p,staticTys) ;
	else {
	  auto statics = getStatics() ;
	  printAux2NodeOp(p,statics,staticTys) ;
	}
      }

      // Finally, inputs and outputs
      auto inputTys = getType().getInputs() ;
      if(isExternal) printAux1NodeOp(p,inputTys) ;
      else {
	auto inputs = getInputs() ;
	printAux2NodeOp(p,inputs,inputTys) ;
      }

      p << " -> ";

      auto outputTypes = getType().getOutputs() ;
      if(isExternal) printAux1NodeOp(p,outputTypes) ;
      else {
	auto outputs = getOutputs() ;
	printAux2NodeOp(p,outputs,outputTypes) ;
      }

      // Now, print the region, if needed
      if(!isExternal) {
	p.printRegion(body,
		      false, // printEntryBlockArgs
		      true   // printBlockTerminators
		      );
      }
    }

    LogicalResult NodeOp::verify() {
      {
	unsigned i = 0;
	for (Type t: getInputsTypes()) {
	  if (!(t.isa<SiginType>())) {
	    return emitOpError()
	      << getNodeName()
	      << "\'s parameter #"
	      << i
	      << " is not a signal.";
	  }
	  i++;
	}
      }
      {
	unsigned i = 0;
	for (Type t: getOutputsTypes()) {
	  if (!(t.isa<SigoutType>())) {
	    return emitOpError()
	      << getNodeName()
	      << "\'s parameter #"
	      << i
	      << " is not a signal.";
	  }
	  i++;
	}
      }
      return success() ;
    }

    Region * NodeOp::getCallableRegion() {
      Region& body = getRegion();
      if(body.empty()) { return NULL ; }
      return &body;
    }

    Region& NodeOp::getBody() {
      Region* region = getCallableRegion() ;
      if(region == NULL) {
	this->emitOpError() << "Body requested on node without regions. "
			    << "Aborting..." ;
	assert(false) ;
      }
      return *region ;
    }

    Operation* NodeOp::getMainLoop() {
      for (Operation &op: *(getBody().begin())) {
	if (isa<scf::ForOp>(&op)) {
	  return &op;
	}
      }
      assert(false);
    }
      
    Block* NodeOp::getMainBlock() {
      Operation *mainLoop = getMainLoop();
      assert(isa<scf::ForOp>(mainLoop));
      scf::ForOp forOp = dyn_cast<scf::ForOp>(mainLoop);
      return &forOp.getLoopBody().front();
    }

    StringRef NodeOp::getNodeName() {
      return
	getOperation()
	->getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
	.getValue();
    }

    iterator_range<Block::args_iterator> NodeOp::getArguments() {
	return getBody().front().getArguments();
      }

    Value NodeOp::getArgument(unsigned i) {
      return getBody().front().getArgument(i);
    }
      
    iterator_range<Block::args_iterator> NodeOp::getStatics() {
      return getBody().front()
	.getArguments()
	.drop_back(getNumInputs() + getNumOutputs());
    }
    
    iterator_range<Block::args_iterator> NodeOp::getInputs() {
      return getBody().front()
	.getArguments()
	.drop_front(getNumStatic())
	.drop_back(getNumOutputs());
    }
    
    iterator_range<Block::args_iterator> NodeOp::getOutputs() {
      return getBody().front()
	.getArguments()
	.drop_front(getNumInputs() + getNumStatic());
    }

    Value NodeOp::getOutput(unsigned i) {
      return getBody().front()
	.getArguments()
	.drop_front(getNumInputs() + getNumStatic() + i).front();
    }
  }
}
