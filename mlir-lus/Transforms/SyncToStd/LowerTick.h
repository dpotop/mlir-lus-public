// -*- C++ -*- //

#ifndef LOWER_TICK_H
#define LOWER_TICK_H

#include <functional>
#include <utility>
#include <list>
#include "mlir/Pass/Pass.h"
#include "../../Dialects/Sync/TickOp.h"

using namespace std;

namespace mlir {
  namespace sync {

    struct LowerTick : public unary_function < ModuleOp, void > {
    public:
      void operator() (ModuleOp moduleOp);
    private:
      FuncOp tickFunc;
      list<Operation*> dishes;
      void tickToCall(OpBuilder &builder, Operation *op);
      void removeSync(Operation *op);
      Operation *buildCall(OpBuilder &builder, Location loc);
    };
  }
}

#endif
