// -*- C++ -*- //

#ifndef MLIRLUS_PARSER_AUX_H
#define MLIRLUS_PARSER_AUX_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <vector>

namespace mlir {

  // Parse a list of arguments and lists of types of the form
  // "(%name1:type1, %name2:type2,...,%nameK:typeK)"
  // or
  // "(type1,...,typeK)"
  // The list can be void, but must always have its parentheses.
  // For a list of types, the result "argNames" is empty.
  // All elements of the list must be of the same kind (simple type
  // or typed arguments).
  ParseResult
    parseArgumentListParen(OpAsmParser &parser,
			   SmallVectorImpl<Type> &argTypes,
			   SmallVectorImpl<OpAsmParser::OperandType> &argNames,
			   SmallVectorImpl<NamedAttrList> &argAttrs) ;

  // Parse a list of types of the form "(type1,...,typeK)".
  // The list can be void.
  // This routine uses the previous one.
  ParseResult
    parseTypeListParen(OpAsmParser &parser,
		       SmallVectorImpl<Type> &argTypes) ;
  
  void printTypeListLoc(OpAsmPrinter &p, ArrayRef<Type> types) ;
}

#endif
