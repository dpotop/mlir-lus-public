#include "WhenOp.h"
#include "KPeriodic.h"

namespace mlir {
  namespace lus {

    Value WhenOp::getCondValue() {
      assert(getOperands().size() == 2) ;
      assert(getCondType().getType() == CondDataType) ;
      return getOperand(1) ;
    }
    
    KPeriodic WhenOp::getCondKPeriodic() {
      assert(getOperands().size() == 1) ;
      assert(getCondType().getType() == CondKPType);
      return getCondType().getWord();
    }
     
    LogicalResult WhenOp::verify() {
      if (isCondData() && !getCondValue().getType().isSignlessInteger(1)) {
	return emitOpError() << "Condition must be boolean, but is "
			     << getCondValue().getType();
      }
      if (getDataType() != getResult().getType()) {
	return emitOpError() << "Operand must be " << getResult().getType()
			     << " but is " << getDataType();
      }
      return success();
    }

    void WhenOp::build(OpBuilder &builder, OperationState &result,
		       Cond<Value> cond, Value input) {
      result.addOperands(input);
      SmallVector<Type, 1> resultTypes;
      resultTypes.push_back(input.getType());
      result.addTypes(resultTypes);
      Cond<Type>* conditionType = NULL ;
      if (cond.getType() == CondDataType) {
	Value data = cond.getData();
	result.addOperands(data);
	assert(data.getType() == builder.getI1Type());
	conditionType = new Cond<Type>(data.getType(), cond.getWhenotFlag());
      }
      else if (cond.getType() == CondKPType) {
	conditionType = new Cond<Type>(cond.getWord());
      }
      else {
	assert(false);
      }
      auto type = builder.getType<WhenType,
				  Type,
				  Cond<Type>>(resultTypes[0],
					      *conditionType) ;
      result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
    }

    ParseResult WhenOp::parse(OpAsmParser &parser, OperationState &result) {
      // This data structure is resposible with the creation of the
      // object.
      auto builder = parser.getBuilder();

      //=======================================================
      // The pure parsing part of the routine

      // First, parse the condition, be it a variable condition, or a
      // K-periodic word.
      bool whenotFlag = false ;
      OpAsmParser::OperandType condValName ;
      KPeriodic* kPeriodic = NULL ;
      {
	// Parse the "not" keyword determining if this is a when or a
	// whenot.
	if (succeeded(parser.parseOptionalKeyword("not"))) {
	  whenotFlag = true ;
	}

	NamedAttrList namedAttrList ;
	if (parser.parseOptionalAttrDict(namedAttrList)) return failure();
	if(namedAttrList.empty()) {
	  if(!succeeded(parser.parseOperand(condValName))) return failure() ;
	} else {
	  Attribute kpAttrRaw = namedAttrList.get("kp") ;
	  assert((bool)kpAttrRaw) ;
	  StringAttr kpAttr = kpAttrRaw.cast<StringAttr>() ;
	  StringRef kpString = kpAttr.getValue() ;
	  kPeriodic = parseKPeriodic(kpString) ;
	}
      }

      // Parse the data argument, which does not have a type
      // (the return type will be used.
      OpAsmParser::OperandType dataValName ;
      Type resultType;
      if (parser.parseOperand(dataValName)
	  || parser.parseColonType(resultType)
	  || parser.resolveOperand(dataValName, resultType, result.operands))
	return failure();
      
      //=======================================================
      // The object building and linking part

      // The return type, which is also needed by the data input.
      result.addTypes(resultType);

      // Create the condition (its type and value objects)
      Cond<Value>* condition = NULL ;
      Cond<Type>* conditionType = NULL ;
      {
	if(kPeriodic == NULL) {
	  // This is a variable-based condition.
	  // The test value has type "i1"
	  auto i1Type = IntegerType::get(resultType.getContext(), 1);
	  // Resolve the condition variable name into a variable,
	  // and place it into the list of operands of the result
	  // (in the second position!).
	  if(parser.resolveOperand(condValName,
				   i1Type,
				   result.operands))
	    return failure();
	  // Based on this 
	  condition = new Cond<Value>(result.operands[1],whenotFlag) ;
	  conditionType = new Cond<Type>(i1Type,whenotFlag) ;
	} else {
	  // Parse k-periodic word
	  condition = new Cond<Value>(*kPeriodic) ;
	  conditionType = new Cond<Type>(*kPeriodic) ;
	}
      }

      // Build the type of the operation and then add it as an
      // attribute to the result before returning
      auto type = builder.getType<WhenType,
				  Type,
				  Cond<Type>>(resultType,
					      *conditionType) ;
      result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
      
      return success();
    }

    void WhenOp::print(OpAsmPrinter &p) {
      p << getOperationName() << " " ;
      
      const WhenType& type = getType() ;
      if(type.getCondType().getType() == CondDataType) {
	if(type.getCondType().getWhenotFlag()) p << "not " ;
	p << getCondValue() << " " ;
      } else {
	p << "{ kp = \"" << type.getCondType().getWord() << "\"} " ;
      }
      
      p << getDataInput() << " " ;

      p << ":" << getDataType() ;
    }

  }
}
