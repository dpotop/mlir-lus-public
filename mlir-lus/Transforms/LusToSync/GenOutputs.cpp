#include "GenOutputs.h"
#include "../../Dialects/Pssa/OutputOp.h"
#include "../../Dialects/Sync/TickOp.h"
#include "../../Dialects/Sync/SyncOp.h"

namespace mlir {
  namespace lus {

    void GenOutputs::operator() (NodeOp nOp) {
      YieldOp yieldOp = nOp.getYield();
      OpBuilder builder = OpBuilder(yieldOp);
      Location loc = yieldOp.getLoc();

      // outputs
      int64_t i = 0;
      SmallVector<Value,4> outputSyncs;
      for (Value v: yieldOp.getOutputs()) {
	pssa::OutputOp outputOp = builder.create<pssa::OutputOp>(loc, i, v);
	outputSyncs.push_back(outputOp.getResult());
	i++;
      }
      
      // tick
      sync::TickOp tickOp = builder.create<sync::TickOp>(yieldOp.getLoc(),
							 outputSyncs);
      // sync
      SmallVector<Value, 4> yieldStates;
      yieldStates.append(yieldOp.getStates().begin(),
			 yieldOp.getStates().end());
      sync::SyncOp syncOp = builder.create<sync::SyncOp>(yieldOp.getLoc(),
							 tickOp.getResult(),
							 yieldStates);

      // yield
      SmallVector<Value, 4> newYieldStates;
      newYieldStates.append(syncOp.getValues().begin(),
			    syncOp.getValues().end());
      SmallVector<Value, 4> yieldOuts;
      yieldOuts.append(yieldOp.getOutputs().begin(),
		       yieldOp.getOutputs().end());
      builder.create<YieldOp>(loc, newYieldStates, yieldOuts);
      yieldOp.erase();
    }
  }
}
