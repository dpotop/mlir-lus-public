#include <iostream>

#include "../../Tools/CommandLine.h"
#include "ClockAnalysis.h"
#include "../../Tools/ParserAux.h"
#include "Node.h"
#include "Yield.h"

namespace mlir {
  namespace lus {

    StringRef NodeOp::getNodeName() {
      return
	getOperation()
	->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
	.getValue();
    }
    
    void NodeOp::build(Builder &builder, OperationState &state,
    		       StringRef name,
    		       ArrayRef<Value> statics,
		       ArrayRef<Value> inputs,
		       ArrayRef<Value> states,
    		       ArrayRef<Type> resultTypes,
    		       RegionKind regionKind) {

      SmallVector<Type,4> staticTypes;
      SmallVector<Type,4> inputTypes;
      SmallVector<Type,4> stateTypes;
      for (Value v : statics) { staticTypes.push_back(v.getType()); }
      for (Value v : inputs) { inputTypes.push_back(v.getType()); }
      for (Value v : states) { stateTypes.push_back(v.getType()); }
      
      state.addAttribute(SymbolTable::getSymbolAttrName(),
    			 builder.getStringAttr(name));
      
      /* auto *body = */ state.addRegion();
      /* auto *body_dom = */ state.addRegion();
      
      auto type = builder.getType<NodeType,
    				  ArrayRef<Type>,
    				  ArrayRef<Type>,
    				  ArrayRef<Type>,
    				  ArrayRef<Type>,
    				  RegionKind> (staticTypes,
					       inputTypes,
					       stateTypes,
    					       resultTypes,
    					       regionKind);
      state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
      
    }
    
    /// Parsing and printing
    ParseResult NodeOp::parse(OpAsmParser &parser, OperationState &result) {

      auto builder = parser.getBuilder();

      // Determine which kind of node is - subject to dominance or not
      RegionKind regionKind = RegionKind::Graph ;
      if (succeeded(parser.parseOptionalKeyword("dom"))) {
	// Dominance is enforced
	regionKind = RegionKind::SSACFG ;
      }

      // Parse the name as a symbol.
      StringAttr nameAttr;
      if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
				 result.attributes)) {
	return failure();
      }

      // Parse the optional static arguments
      SmallVector<OpAsmParser::OperandType, 4> staticNames;
      SmallVector<Type, 4> staticTypes;
      SmallVector<NamedAttrList,4> staticAttrs; // Currently unused
      if (succeeded(parser.parseOptionalKeyword("static"))) {
	// There is a list of static arguments
	if (parseArgumentListParen(parser,
				   staticTypes,staticNames,staticAttrs)){
	  return failure();
	}
      }

      // Parse the optional state arguments
      SmallVector<OpAsmParser::OperandType, 4> stateNames;
      SmallVector<Type, 4> stateTypes;
      SmallVector<NamedAttrList,4> stateAttrs; // Currently unused
      if (succeeded(parser.parseOptionalKeyword("state"))) {
	// There is a list of static arguments
	if (parseArgumentListParen(parser,stateTypes,stateNames,stateAttrs)){
	  return failure();
	}
      }

      // Parse the input arguments. The list must be present, even
      // if it is empty.
      SmallVector<OpAsmParser::OperandType, 4> inputNames;
      SmallVector<Type, 4> inputTypes;
      SmallVector<NamedAttrList,4> inputAttrs; // Currently unused
      if (parseArgumentListParen(parser, inputTypes, inputNames, inputAttrs)){
	return failure();
      }
      
      // This is not optional, even if there's no output
      if (parser.parseArrow()) {
	return failure();
      }
      
      // Parse the result types. The list must be present, even
      // if it is empty.
      SmallVector<Type, 4> resultTypes;
      if (parseTypeListParen(parser,resultTypes)){
	return failure();
      }
      
      SmallVector<OpAsmParser::OperandType, 4> regionArgs;
      regionArgs.append(staticNames.begin(), staticNames.end());
      regionArgs.append(inputNames.begin(), inputNames.end());
      regionArgs.append(stateNames.begin(), stateNames.end());
      SmallVector<Type, 4> regionArgsTypes;
      regionArgsTypes.append(staticTypes.begin(), staticTypes.end());
      regionArgsTypes.append(inputTypes.begin(), inputTypes.end());
      regionArgsTypes.append(stateTypes.begin(), stateTypes.end());
      
      // Actual parsing of the body. We use here two regions, the one without
      // dominance, the second with dominance enforced. Note that these two
      // regions exist always, even if they may be empty.
      auto *body = result.addRegion();
      auto *body_dom = result.addRegion();
      switch(regionKind){
      case RegionKind::Graph:
	if (!parser.parseOptionalRegion(*body, regionArgs, regionArgsTypes).hasValue()) {
	  return failure() ;
	}
	break ;
      case RegionKind::SSACFG:
	if (!parser.parseOptionalRegion(*body_dom,
					regionArgs, regionArgsTypes).hasValue()) {
	  return failure() ;
	}
	break;
      default: assert(false) ;
      }
      // Build the NodeType of the node, and add it as an attribute to
      // the node.
      auto type = builder.getType<NodeType,
				  ArrayRef<Type>,
				  ArrayRef<Type>,
				  ArrayRef<Type>,
				  ArrayRef<Type>,
				  RegionKind>(staticTypes, inputTypes,
					      stateTypes,
					      resultTypes,regionKind);

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
	auto val = get<0>(pr) ;
	auto typ = get<1>(pr) ;
	if (i > 0) p << ", " ; i++ ;
	p.printOperand(val) ;
	p << ":" ;
	p.printType(typ) ;
      }
      p << ')';	
    }
		  
    void NodeOp::print(OpAsmPrinter &p) {
      // print the node name  
      StringRef nodeName = getNodeName() ;
      p << getOperationName() << ' ';

      if(getType().getRegionKind() == RegionKind::SSACFG) {
	// Dominance is enforced
	p << " dom " ;
      }
      
      p.printSymbolName(nodeName);
      
      // Print the signature inputs  
      Region &body = this->getActiveRegion();
      bool isExternal = body.empty();

      auto staticTypes = getType().getStatics() ;
      if(staticTypes.size() != 0) {
	p << " static " ;
	if(isExternal) printAux1NodeOp(p,staticTypes) ;
	else {
	  auto statics = getStatics() ;
	  printAux2NodeOp(p,statics,staticTypes) ;
	}
      }

      auto stateTypes = getType().getStates() ;
      if(stateTypes.size() != 0) {
	p << " state " ;
	if(isExternal) printAux1NodeOp(p,stateTypes) ;
	else {
	  auto states = getStates() ;
	  printAux2NodeOp(p,states,stateTypes) ;
	}
      }

      // Finally, inputs and outputs
      auto inputTypes = getType().getInputs() ;
      if(isExternal) printAux1NodeOp(p,inputTypes) ;
      else {
	auto inputs = getInputs() ;
	printAux2NodeOp(p,inputs,inputTypes) ;
      }
      p << " -> ";
      auto resultTypes = getType().getResults() ;
      printAux1NodeOp(p,resultTypes) ;

      // Now, print the region, if needed
      if(!isExternal) {
	p.printRegion(body,
		      false, // printEntryBlockArgs
		      true   // printBlockTerminators
		      );
      }
    }

    LogicalResult NodeOp::verify() {
      
      if(getCallableRegion() != NULL) {

	Operation* terminatorOp = getBody().back().getTerminator();
	if (!isa<YieldOp>(terminatorOp)) {
	  return emitOpError() << "An operation yield "
			       << "must terminate a node.";
	}
	
	YieldOp yieldOp = dyn_cast<YieldOp>(terminatorOp);
	if (getNumOutputs() != yieldOp.getNumOutputs()) {
	  return emitOpError()
	    << getNodeName() << " has " << getNumOutputs() << " outputs but "
	    << "its yield has " << yieldOp.getNumOutputs();
	}
	if (getNumStates() != yieldOp.getNumStates()) {
	  return emitOpError()
	    << getNodeName() << " has " << getNumStates() << " states but "
	    << "its yield has " << yieldOp.getNumStates();
	}
	{
	  unsigned i = 0;
	  for (auto e : llvm::zip(getOutputsTypes(),
				  yieldOp.getOutputsTypes())) {
	    if (get<0>(e) != get<1>(e)) {
	      return emitOpError()
		<< getNodeName() << "'s #" << i << " output is " << get<0>(e)
		<< " but its yield's #" << i << " output is " << get<1>(e);
	    }
	    i++;
	  }
	}
	{
	  unsigned i = 0;
	  for (auto e : llvm::zip(getStatesTypes(),
				  yieldOp.getStatesTypes())) {
	    if (get<0>(e) != get<1>(e)) {
	      return emitOpError()
		<< getNodeName() << "'s #" << i << " state is " << get<0>(e)
		<< " but its yield's #" << i << " state is " << get<1>(e);
	    }
	    i++;
	  }
	}
      }
      
      if (!disableClockAnalysis) {
	ClockAnalysis ca(*this);
	if(!succeeded(ca.analyse())) return failure() ;
      }
      
      return success() ;
    }

    RegionKind NodeOp::getRegionKind(unsigned index) {
	switch(index){
	case 0: return RegionKind::Graph ;
	case 1: return RegionKind::SSACFG ;
	default: assert(false) ;
	}
      }

    Region * NodeOp::getCallableRegion() {
      Region& body = getActiveRegion() ;
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

    bool NodeOp::isDominanceFree() {
	return getType().getRegionKind() == RegionKind::Graph;
      }
    
    void NodeOp::forceDominance() {
      // Change type
      assert(getType().getRegionKind() == RegionKind::Graph);
      Region& graphRegion = getActiveRegion();
      auto type = NodeType::get(getContext(),
    				getStaticTypes(),
    				getInputsTypes(),
    				getStatesTypes(),
    				getCallableResults(),
    				RegionKind::SSACFG);
      getOperation()->setAttr(getTypeAttrName(), TypeAttr::get(type));
      Region& SSACFGRegion = getActiveRegion();
      SSACFGRegion.takeBody(graphRegion);
    }

    Region& NodeOp::getActiveRegion() {
	auto regions = this->getOperation()->getRegions();
	assert(regions.size() == 2) ;
	switch(getType().getRegionKind()){
	case RegionKind::Graph: return regions[0] ;
	case RegionKind::SSACFG:return regions[1] ;
	default: assert(false&&"Unknown RegionKind in a NodeOp.") ;
	}
      }

    iterator_range<Block::args_iterator> NodeOp::getStatics() {
      return
	getActiveRegion().front()
	.getArguments().drop_back(getNumInputs()+getNumStates());
    }
    
    iterator_range<Block::args_iterator> NodeOp::getInputs() {
      return
	getActiveRegion().front()
	.getArguments().drop_front(getNumStatic()).drop_back(getNumStates());
    }
    
    iterator_range<Block::args_iterator> NodeOp::getStates() {
      return
	getActiveRegion().front()
	.getArguments().drop_front(getNumStatic()+getNumInputs());
      }

    Value NodeOp::addState(Type stateType) {
      Value state = getBody().addArgument(stateType);
      SmallVector<Type, 4> statesTypes;
      for (Type t : getStatesTypes())
	statesTypes.push_back(t);
      statesTypes.push_back(stateType);
      auto type = NodeType::get(getContext(),
				getStaticTypes(),
				getInputsTypes(),
				statesTypes,
				getCallableResults(),
				getType().getRegionKind());
      getOperation()->setAttr(getTypeAttrName(), TypeAttr::get(type));
      return state;
    }

    YieldOp NodeOp::getYield() {
      Operation* termOp = getBody().back().getTerminator();
      assert(isa<YieldOp>(termOp));
      YieldOp yieldOp = dyn_cast<YieldOp>(termOp);
      return yieldOp;
    }
    
  }
}
