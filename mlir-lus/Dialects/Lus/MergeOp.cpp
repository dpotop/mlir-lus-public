#include "MergeOp.h"

namespace mlir {
  namespace lus {
    
    Value MergeOp::getCondValue() {
      assert(getOperands().size() == 3) ;
      assert(getCondType().getType() == CondDataType) ;
      return getOperand(2) ;
    }
    
    KPeriodic MergeOp::getCondKPeriodic() {
      assert(getOperands().size() == 2) ;
      assert(getCondType().getType() == CondKPType);
      return getCondType().getWord();
    }

    void MergeOp::build(OpBuilder &builder, OperationState &result,
			Cond<Value> cond, Value trueInput, Value falseInput) {
      Type inputType = trueInput.getType();
      result.addOperands(trueInput);
      result.addOperands(falseInput);
      result.addTypes({inputType});
      Cond<Type>* conditionType = NULL ;
      if (cond.getType() == CondDataType) {
	Value data = cond.getData();
	result.addOperands(data);
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
				  Cond<Type>>(inputType, *conditionType) ;
      result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
    }

    ParseResult MergeOp::parse(OpAsmParser &parser,
			       OperationState &result) {
      // This data structure is resposible with the creation of the
      // object.
      auto builder = parser.getBuilder();

      //=======================================================
      // The pure parsing part of the routine

      // First, parse the condition, be it a variable condition, or a
      // K-periodic word.
      OpAsmParser::OperandType condValName ;
      KPeriodic* kPeriodic = NULL ;
      {
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

      // Parse the data arguments, which does not have a type
      // (the return type will be used.
      OpAsmParser::OperandType trueValName ;
      OpAsmParser::OperandType falseValName ;
      Type resultType;
      if (parser.parseOperand(trueValName)
	  || parser.parseOperand(falseValName)
	  || parser.parseColonType(resultType)
	  || parser.resolveOperand(trueValName, resultType, result.operands)
	  || parser.resolveOperand(falseValName,resultType,result.operands))
	return failure();

      result.addTypes({resultType});

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
	  condition = new Cond<Value>(result.operands[1],false) ;
	  conditionType = new Cond<Type>(i1Type,false) ;
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
				  Cond<Type>>(resultType, *conditionType) ;
      result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
      
      return success();

    }
    
    LogicalResult MergeOp::verify() {
      Type trueType = getTrueInput().getType();
      Type falseType = getFalseInput().getType();
      Type resultType = getResult().getType();
      if (trueType != falseType) {
	return emitOpError()
	  << "Inputs should have the same type but "
	  << "true branch is " << trueType << " and "
	  << "false branch is " << falseType;
      }
      if (trueType != resultType) {
	return emitOpError()
	  << "Inputs and result should have the same type but "
	  << "inputs are " << trueType << " and "
	  << "result is " << resultType;
      }
      if (isCondData() && !getCondValue().getType().isSignlessInteger(1)) {
	return emitOpError() << "Condition must be boolean, but is "
			     << getCondValue().getType();
      }
      return success();
    }

    void MergeOp::print(OpAsmPrinter &p) {
      p << getOperationName() << " " ;
      
      const WhenType& type = getType() ;
      if(type.getCondType().getType() == CondDataType) {
	assert(!type.getCondType().getWhenotFlag()) ;
	p << getCondValue() << " " ;
      } else {
	p << "{ kp = \"" << type.getCondType().getWord() << "\"} " ;
      }
      
      p << getTrueInput() << " " ;
      p << getFalseInput() << " " ;

      p << ":" << getDataType() ;
    }
  }
}
