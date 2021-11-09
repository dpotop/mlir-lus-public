// -*- C++ -*- //

#ifndef LOWER_PSSA_H
#define LOWER_PSSA_H

#include <functional>
#include <utility>
#include <list>
#include <unordered_map>
#include "mlir/Pass/Pass.h"
#include "../../Dialects/Sync/Node.h"
#include "../../Dialects/Pssa/CreatePredOp.h"
#include "../../Dialects/Pssa/CondactOp.h"
#include "../../Dialects/Pssa/OutputOp.h"
#include "../Utilities/ValueHash.h"
#include "../Utilities/ConstantPool.h"

using namespace mlir::sync;

namespace mlir {
  namespace pssa {

    struct LowerPssa : public unary_function < sync::NodeOp, void > {
    public:
      void operator() (sync::NodeOp nodeOp);
    private:
      ConstantPool *constantPool;
      list<Operation*> dishes;
      void lower(sync::NodeOp nodeOp, Operation *op);
      void lowerCreatePredOp(CreatePredOp createPredOp);
      void lowerCondactOp(CondactOp condactOp);
      void lowerYieldOp(YieldOp yieldOp);
      void lowerOutputOp(NodeOp nodeOp, OutputOp outputOp);
    };
  }
}

#endif
