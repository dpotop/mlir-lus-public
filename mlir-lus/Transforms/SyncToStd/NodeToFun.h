// -*- C++ -*- //

#ifndef SYNC_NODE_TO_FUN_H
#define SYNC_NODE_TO_FUN_H

#include <functional>
#include <utility>
#include <list>
#include <unordered_set>
#include "mlir/Pass/Pass.h" // For ModuleOp
#include "../../Dialects/Sync/Node.h"
#include "../../Dialects/Sync/InstOp.h"
#include "../../Dialects/Sync/InputOp.h"
#include "../../Dialects/Sync/OutputOp.h"
#include "../Utilities/ExtFunctionPool.h"
#include "../Utilities/ConstantPool.h"

using namespace std;

namespace mlir {
  namespace sync {

    struct NodeToFun : public unary_function < ModuleOp, void > {
    public:
      void operator() (ModuleOp moduleOp);
    private:
      ExtFunctionPool *extFunctionPool;
      unordered_set<Operation*> instances;
      list<Operation*> dishes;

      void instantiateNodes(Operation *op);
      void instantiate(InstOp instOp);
      
      FuncOp buildStartFun(InstOp instOp);

      FuncOp buildInstFun(InstOp instOp, FuncOp startFun);

      FuncOp buildCoreFun(InstOp instOp);
      FuncOp buildCoreFun(NodeOp nodeOp);
      
      void lowerIO(FuncOp funcOp, Operation *op, unsigned offset_out);
      Operation* lowerAndBufferizeInput(InputOp inputOp, FuncOp funcOp,
					ConstantPool &cp);
      Operation* lowerAndBufferizeOutput(OutputOp outputOp, FuncOp funcOp,
					 ConstantPool &cp,
					 unsigned offset_out);

      void instOpToCall(InstOp instOp, FuncOp toBeCalled);

      
    };
  }
}

#endif
