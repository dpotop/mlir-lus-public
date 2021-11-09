#include <iostream>
#include "CommandLine.h" // For verboseLevel
#include "ParserAux.h"


namespace mlir {
  
  // Parse a non-void list of arguments and lists of types of the form
  // "%name1:type1, %name2:type2,...,%nameK:typeK"
  // or
  // "type1,...,typeK"
  // For a list of types, the result "argNames" is empty.
  // The argument list either has to consistently have ssa-id's
  // followed by types, or just be a type list.  It isn't ok to
  // sometimes have SSA ID's and sometimes not.
  // Error messages include location.
  ParseResult
  parseNVArgumentList(OpAsmParser &parser,
		      SmallVectorImpl<Type> &argTypes,
		      SmallVectorImpl<OpAsmParser::OperandType> &argNames,
		      SmallVectorImpl<NamedAttrList> &argAttrs) {
    // Lambda function that reads one argument of the form
    // 
    auto parseArgument = [&]() -> ParseResult {
			   llvm::SMLoc loc = parser.getCurrentLocation();
      
			   // Parse argument name if present.
			   OpAsmParser::OperandType argument;
			   Type argumentType;
			   if (succeeded(parser.parseOptionalRegionArgument(argument)) &&
			       !argument.name.empty()) {
			     // Reject this if the preceding argument was missing a name.
			     if (argNames.empty() && !argTypes.empty())
			       return parser.emitError(loc, "expected type instead of SSA identifier");
			     argNames.push_back(argument);
			     // If there is an SSA identifier, there must also be a colon,
			     // and then a type.
			     if (parser.parseColonType(argumentType)) return failure();
			   } else if (!argNames.empty()) {
			     // Reject this if the preceding argument had a name.
			     return parser.emitError(loc, "expected SSA identifier");
			   } else if (parser.parseType(argumentType)) {
			     // This is the case where only a type is provided.
			     return failure();
			   }
      
			   // In all cases, add the argument type.
			   argTypes.push_back(argumentType);
      
			   // Parse any argument attributes.
			   NamedAttrList attrs;
			   if (parser.parseOptionalAttrDict(attrs))
			     return failure();
			   argAttrs.push_back(attrs);
			   return success();
			 };
    
    // Actual parsing, using the previous lambda function.
    do {
      if (parseArgument()) return failure();
    } while (succeeded(parser.parseOptionalComma()));
    
    return success();
  }

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
			 SmallVectorImpl<NamedAttrList> &argAttrs) {
    if (parser.parseLParen()) return failure();
    
    // Actual parsing, using the previous lambda function.
    if (failed(parser.parseOptionalRParen())) {
      if(parseNVArgumentList(parser,argTypes,argNames,argAttrs))
	return failure();
      parser.parseRParen();
    }
    
    return success();
  }

  
  // Using the previous routine, specialization that only
  // parses a type list
  ParseResult
  parseTypeListParen(OpAsmParser &parser,
		     SmallVectorImpl<Type> &argTypes) {
    // To call parseArgumentList I have to provide a vector of
    // names and one of attributes. They will not be used.
    SmallVector<OpAsmParser::OperandType, 1> unusedNames;
    SmallVector<NamedAttrList,1> unusedAttrs;
    if (parseArgumentListParen(parser,argTypes,unusedNames,unusedAttrs))
      return failure();
    // TODO : I will comment the following line. I wonder if it's correct...
    assert(unusedNames.size() == 0) ;
    // TODO: Here, I'd like to assert that unusedAttrs is empty, but
    // this is not true. In the typical case, it contains one element,
    // which is an empty NamedAttrList... Maybe report this to the
    // MLIR forum as a bug.
    return success() ;
  }
  
  // Print a type list, Lus style (parentheses always present)
  void printTypeListLoc(OpAsmPrinter &p, ArrayRef<Type> types) {
    p << '(';
    llvm::interleaveComma(types,p);
    p << ')';
  }
  
}
