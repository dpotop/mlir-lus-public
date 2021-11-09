// -*- C++ -*- //

#ifndef REMOVE_PSIS_H
#define REMOVE_PSIS_H

#include <functional>
#include <list>
#include <unordered_map>
#include "../Utilities/ValueHash.h"
#include "../Utilities/ConstantPool.h"

namespace mlir {
  namespace pssa {

    struct RemovePsis : public std::unary_function < ModuleOp, void > {
    public:
      void operator() (ModuleOp moduleOp);
    private:
      std::unordered_map<Type, FuncOp, TypeHash> selectFuncs;
      std::list<Operation*> dishes;
      ModuleOp moduleOp;
      void apply(Operation *op, ConstantPool &constantPool);
    };
  }
}

#endif
