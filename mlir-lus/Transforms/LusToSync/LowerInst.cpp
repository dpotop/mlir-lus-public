#include "LowerInst.h"

namespace mlir {
  namespace sync {

    int64_t LowerInst::nextId = 2;
    
    void LowerInst::lower(Operation *op) {

      for (Region &r: op->getRegions()) {
	if (!r.empty()) {
	  for (Operation &deeperOp: *r.begin()) {
	    lower(&deeperOp);
	  }
	}
      }
      
      if (isa<LusInstOp>(op)) {
	LusInstOp lusInstOp = dyn_cast<LusInstOp>(op);
	int64_t id = getId();
	StringRef calleeName = lusInstOp.getCalleeName();
	SmallVector<Value, 4> params;
	params.append(lusInstOp.getArgOperands().begin(),
		      lusInstOp.getArgOperands().end());
	SmallVector<Type, 4> resultTypes;
	resultTypes.append(lusInstOp.getResults().getTypes().begin(),
			   lusInstOp.getResults().getTypes().end());

	OpBuilder builder(lusInstOp);
	OperationState state(lusInstOp.getLoc(),
			     SyncInstOp::getOperationName());
	SyncInstOp::build(builder, state,
			  calleeName, id, params, resultTypes);
	Operation *syncInstOpPtr = builder.createOperation(state);
	op->replaceAllUsesWith(syncInstOpPtr);
	dishes.push_back(op);
      }
    }

    void LowerInst::operator() (ModuleOp moduleOp) {
      lower(moduleOp);
      for (Operation *op : dishes) {
	op->erase();
      }
    }
  }
}
