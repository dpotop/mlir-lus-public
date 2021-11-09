#include "CreatePredOp.h"

namespace mlir {
  namespace pssa {

    Value CreatePredOp::data() {
      assert(isDataDependent());
      return getOperand(0);
    }

    Value CreatePredOp::getComplementPred(Value v) {
      if (v == getTruePred()) { return getFalsePred(); }
      else if (v == getFalsePred()) { return getTruePred(); }
      else { assert(false); }
    }

    KPeriodic CreatePredOp::word() {
      assert(isKPeriodic());
      return getType().getWord();
    }

    void CreatePredOp::build(Builder *builder,
			     OperationState &odsState,
			     Value complement) {
      odsState.addOperands(complement);
      odsState.addTypes(builder->getI1Type());
      odsState.addTypes(builder->getI1Type());
      // TODO whenot flag : useless, Cond should be simpler
      Cond<Type> condType(complement.getType(), false);
      auto type = builder->getType<CreatePredType,
				   Cond<Type>> (condType);
      odsState.addAttribute(getTypeAttrName(), TypeAttr::get(type));
    }

    void CreatePredOp::build(Builder *builder,
			     OperationState &odsState,
			     KPeriodic word) {
      odsState.addTypes(builder->getI1Type());
      odsState.addTypes(builder->getI1Type());
      // TODO whenot flag : useless, Cond should be simpler
      Cond<Type> condType(word);
      auto type = builder->getType< CreatePredType,
				    Cond<Type>> (condType);
      odsState.addAttribute(getTypeAttrName(), TypeAttr::get(type));
    }

    ParseResult CreatePredOp::parse(OpAsmParser &parser,
				    OperationState &result) {

      auto builder = parser.getBuilder();
      
      OpAsmParser::OperandType op;
      Type type1;
      Type type2;
      
      if (parser.parseOperand(op) ||
	  parser.parseOptionalAttrDict(result.attributes) ||
	  parser.resolveOperand(op, parser.getBuilder().getI1Type(),
				result.operands) ||
	  parser.parseColonType(type1) ||
	  parser.parseComma() ||
	  parser.parseType(type2))
	return failure();

      parser.addTypeToList(type1, result.types);
      parser.addTypeToList(type2, result.types);

      // whenot flag : useless, Cond should be simpler
      Cond<Type> condType(result.operands[0].getType(), false);
      auto type = builder.getType< CreatePredType,
				   Cond<Type>> (condType);
      result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
      
      return success();
    }
    
    void CreatePredOp::print(OpAsmPrinter &p) {
      p << getOperationName() << ' ';
      if (isDataDependent()) p << data() ;
      else if (isKPeriodic()) p << "\"" << word() << "\"";
      else assert(false);
      // p.printOptionalAttrDict(getAttrs());
      p << " : "
	<< getResult(0).getType()
	<< ", "
	<< getResult(1).getType();
    }
      
    LogicalResult CreatePredOp::verify() {
      OpBuilder builder(getOperation());
      if (getTruePred().getType() != builder.getI1Type()
	  || getFalsePred().getType() != builder.getI1Type()) {
	return emitOpError() << "Both two predicates must be boolean.";
      }
      return mlir::success();
    }

  }
}
