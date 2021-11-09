#include "Yield.h"
#include "Node.h"
#include "../../Tools/ParserAux.h"

namespace mlir {
  namespace lus {

    void YieldOp::build(Builder &builder, OperationState &state,
			ArrayRef<Value> states, ArrayRef<Value> outputs) {
      
      SmallVector<Type,4> outputTypes;
      SmallVector<Type,4> stateTypes;
      for (Value v : outputs) { outputTypes.push_back(v.getType()); }
      for (Value v : states) { stateTypes.push_back(v.getType()); }

      state.addOperands(states);
      state.addOperands(outputs);

      // Add the YieldType
      auto type = builder.getType
	<YieldType, ArrayRef<Type>,ArrayRef<Type>>
	(stateTypes, outputTypes);
      state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
    }

    /// The lus.yield operation
    ParseResult YieldOp::parse(OpAsmParser &parser, OperationState &result) {
      
      auto builder = parser.getBuilder();
      llvm::SMLoc loc = parser.getCurrentLocation();

      // Parse the state and output lists (value names and types)
      SmallVector<OpAsmParser::OperandType, 4> stateNames;
      SmallVector<Type, 4> stateTypes;
      SmallVector<OpAsmParser::OperandType, 4> outNames;
      SmallVector<Type, 4> outTypes;
      SmallVector<NamedAttrList,4> stateAttrs; // Currently unused
      SmallVector<NamedAttrList,4> outAttrs; // Currently unused

      // States
      if (succeeded(parser.parseOptionalKeyword("state"))) {
	// There is a list of static arguments
	if (parseArgumentListParen(parser,stateTypes,stateNames,stateAttrs)){
	  return failure();
	}
      }

      // Outputs
      if(parseArgumentListParen(parser,outTypes,outNames,outAttrs)){
	return failure();
      }

      // Both states and outputs are stored on the operand list
      // of the operator
      SmallVector<OpAsmParser::OperandType, 4> operandNames;
      SmallVector<Type, 4> operandTypes;
      operandNames.append(stateNames.begin(),stateNames.end());
      operandNames.append(outNames.begin(),outNames.end());
      operandTypes.append(stateTypes.begin(),stateTypes.end());
      operandTypes.append(outTypes.begin(),outTypes.end());

      if (parser.resolveOperands(operandNames, operandTypes, loc,
				 result.operands)){
	return failure();
      }

      // Add the type
      auto type = builder.getType<YieldType,
				  ArrayRef<Type>,
				  ArrayRef<Type>>(stateTypes, outTypes);
      result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

      return success();
    }

    void YieldOp::print(OpAsmPrinter &p) {
      p << getOperationName() << " " ; 
      
      auto stateValues = getStates() ;
      auto stateTypes = getStatesTypes() ;
      auto outValues = getOutputs();
      auto outTypes = getOutputsTypes() ;
      assert(stateValues.size() == stateTypes.size()) ;
      assert(outValues.size() == outTypes.size()) ;
      
      if (stateValues.size() > 0) {
	p << "state (";
	for(int i=0;i<stateValues.size();i++) {
	  if (i > 0) p << ", ";
	  p.printOperand(stateValues[i]);
	  p << ":" ;
	  p.printType(stateTypes[i]) ;
	}
	p << ")";
      }
      {
	p << " (";
	for(int i=0;i<outValues.size();i++) {
	  if (i > 0) p << ", ";
	  p.printOperand(outValues[i]);
	  p << ":" ;
	  p.printType(outTypes[i]) ;
	}
	p << ")";
      }
    }

    LogicalResult YieldOp::verify() {
      
      // The parent op should be a Node of some kind, which has a symbol
      // name. This line will test whether this name exists.
      // Types correspondance between this yield and its parent node is
      // checked in NodeOp verification.
      
      if(!isa<NodeOp>(getOperation()->getParentOp())) {
	// Only the new node types are covered
	return emitOpError() << "Verification failure: "
			     << getOperationName() << " only terminates "
      			     << NodeOp::getOperationName() ;
      }
      return success();
    }

    Operation::operand_range YieldOp::getStates() {
      return getOperands().drop_back(getNumOutputs());
    }
    
    Operation::operand_range YieldOp::getOutputs() {
      return getOperands().drop_front(getNumStates());
    }
    
    void YieldOp::addState(Value state) {
      getOperation()->insertOperands(getNumStates(), state);
      // Rebuild YieldType
      SmallVector<Type, 4> statesTypes;
      for (Type t : getStatesTypes())
	statesTypes.push_back(t);
      statesTypes.push_back(state.getType());
      auto type = YieldType::get(getContext(),
				 statesTypes,
				 getOutputsTypes());
      getOperation()->setAttr(getTypeAttrName(), TypeAttr::get(type));
    }
  }
}
