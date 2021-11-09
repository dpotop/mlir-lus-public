// -*- C++ -*- //

#ifndef NODE_TO_NODE_H
#define NODE_TO_NODE_H

#include <list>
#include <functional>
#include "mlir/Pass/Pass.h" // For ModuleOp
#include "../../Dialects/Lus/Node.h"
#include "../../Dialects/Sync/Node.h"
#include "..//Utilities/ConstantPool.h"

namespace mlir {
  namespace sync {
    struct NodeToNode: public std::unary_function <ModuleOp, void> {
    private:
      using LusNodeOp = lus::NodeOp;
      using SyncNodeOp = sync::NodeOp;
    public:
      void operator() (ModuleOp moduleOp);
    private:
      std::list<Operation*> dishes;
      std::list<Block*> dishesBlock;
      // The whole lowering from lus::NodeOp to sync::NodeOp
      SyncNodeOp lower(LusNodeOp lusNodeOp);
      // Build sync::NodeOp skeleton
      SyncNodeOp buildSkeleton(LusNodeOp lusNodeOp);
      // Update block parameters in the loop
      void updateLoopParameters(Block *loopBlock, SyncNodeOp syncNodeOp);
      // Lower yield at the end of the loop
      void lowerYieldOp(lus::YieldOp, SyncNodeOp);
      void lowerInputs(SyncNodeOp syncNodeOp, LusNodeOp lusNodeOp);
      void prepareKperiodicWord(ConstantPool &constantPool,
				OpBuilder &builder,
				Block *nodeBlock,
				std::list<Value> &inits,
				std::list<Value> &update,
				Operation *op);
    };
  }
}

#endif
