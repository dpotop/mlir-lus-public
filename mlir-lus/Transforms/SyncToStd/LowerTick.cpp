#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "LowerTick.h"
#include "../../Dialects/Sync/Sync.h"
#include "../../Dialects/Sync/SyncOp.h"
#include "../../Dialects/Sync/HaltOp.h"

namespace mlir {
  namespace sync {
    void LowerTick::operator() (ModuleOp moduleOp) {

      OpBuilder builder(&moduleOp.getBodyRegion());
      Location loc = moduleOp.getLoc();

      FunctionType tickFT = builder.getFunctionType({},
						    builder.getI32Type());
      OperationState tickState(loc, FuncOp::getOperationName());
      StringAttr visibility = builder.getStringAttr("private") ;
      FuncOp::build(builder,
		    tickState,
		    StringAttr::get(builder.getContext(),
				    TickOp::getFunctionName()),
		    TypeAttr::get(tickFT),
		    visibility
		    );
      
      Operation *tickFuncOpPtr = builder.createOperation(tickState);
      tickFunc = dyn_cast<FuncOp>(tickFuncOpPtr);
      
      tickToCall(builder, moduleOp.getOperation());
      removeSync(moduleOp.getOperation());
      for (Operation *op : dishes) {
	op->erase();
      }
    }

    void LowerTick::tickToCall(OpBuilder &builder, Operation *op) {
      for (Region &r : op->getRegions()) {
	for (Operation &innerOp : r.getOps()) {
	  tickToCall(builder, &innerOp);
	}
      }
      if (isa<TickOp>(op)) {
	TickOp tickOp = dyn_cast<TickOp>(op);
	Location loc = tickOp.getLoc();
	builder.setInsertionPoint(op);
	Operation *callOpPtr = buildCall(builder, loc);
	tickOp.replaceAllUsesWith(callOpPtr);
	dishes.push_back(tickOp);
      }
      else if (isa<HaltOp>(op)) {

	HaltOp haltOp = dyn_cast<HaltOp>(op);
	Location loc = haltOp.getLoc();
	builder.setInsertionPoint(op);

	OperationState state(loc, ReturnOp::getOperationName());
	ReturnOp::build(builder,state);
	builder.createOperation(state);
	
	dishes.push_back(op);
      }
    }

    void LowerTick::removeSync (Operation *op) {
      for (Region &r : op->getRegions()) {
	for (Operation &innerOp : r.getOps()) {
	  removeSync(&innerOp);
	}
      }
      if (isa<SyncOp>(op)) {
	SyncOp syncOp = dyn_cast<SyncOp>(op);
	OperandRange values = syncOp.getValues();
	for (auto e: llvm::zip(syncOp.getResults(), values)) {
	  get<0>(e).replaceAllUsesWith(get<1>(e));
	}
	dishes.push_back(op);
      }
    }
    
    Operation *LowerTick::buildCall(OpBuilder &builder, Location loc) {
      OperationState callState(loc, CallOp::getOperationName());
      CallOp::build(builder, callState, tickFunc);
      Operation *callOpPtr = builder.createOperation(callState);
      return callOpPtr;
    }
  }
}
