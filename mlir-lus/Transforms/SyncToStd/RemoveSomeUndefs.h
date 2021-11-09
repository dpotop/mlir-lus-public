// -*- C++ -*- //

#ifndef REMOVE_SOME_UNDEFS_H
#define REMOVE_SOME_UNDEFS_H

#include <functional>
#include <list>
#include <unordered_map>
// #include "mlir/Pass/Pass.h" // For ModuleOp
// #include "mlir/IR/Module.h"
// #include "mlir/IR/Function.h"
#include "../Utilities/ValueHash.h"

namespace mlir {
  namespace pssa {

    struct RemoveSomeUndefs : public std::unary_function < ModuleOp, void > {
    public:
      void operator() (ModuleOp moduleOp);
    private:
      std::unordered_map<Type, FuncOp, TypeHash> undefFuncs;
      std::unordered_map<Type, FuncOp, TypeHash> selectFuncs;
      std::list<Operation*> dishes;
      ModuleOp moduleOp;
      void apply(Operation *op);
    };
  }
}

#endif
