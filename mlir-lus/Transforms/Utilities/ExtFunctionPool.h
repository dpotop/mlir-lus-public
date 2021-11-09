// -*- C++ -*- //

#ifndef EXT_FUNCTION_POOL_H
#define EXT_FUNCTION_POOL_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "ValueHash.h"
#include <unordered_map>
#include <vector>

namespace mlir {

  struct ExtFunctionPool {
  public:
    ExtFunctionPool(ModuleOp m): myBuilder(m.body()), mod(m) {}
    FuncOp schStart(Type t);
    FuncOp schSetInstance();
    FuncOp schSetIOInput(Type t);
    FuncOp schSetIOOutput(Type t);
    FuncOp inst();
    FuncOp build(std::string name, FunctionType ft);
  private:
    OpBuilder myBuilder;
    ModuleOp mod;
    std::unordered_map<std::string, Operation*> funcOps;
    std::string name(std::string prefix, FunctionType ft);
    std::string printable(Type t);
  };

}

#endif
