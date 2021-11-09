#include "RemovePre.h"
#include <vector>
#include "../../Dialects/Lus/PreOp.h"
#include "../../Dialects/Lus/WhenOp.h"

namespace mlir {
  namespace lus {

    void RemovePre::operator() (NodeOp nop) {
      nodeOp = nop;
      for (Operation& op : *(nop.getBody().begin())) {
	removePre(&op);
      }
      for (Operation* op : former) {
	op->erase();
      }
    }
    
    void RemovePre::removePre(Operation* op) {

      // apply in nested regions
      for (Region& r : op->getRegions()) {
	for (Operation& innerOp : *(r.begin())) {
	  removePre(&innerOp);
	}
      }
      
      if (isa<PreOp>(op)) {
	
	PreOp preOp = dyn_cast<PreOp>(op);
	OpBuilder builder(op);
	Location loc = op->getLoc();

	// If dominance is not granted in the node :
	// the pre result should become an explicit state.
	// It makes possible to enforce dominance later.
	if (nodeOp.isDominanceFree()) {

	  // Instead of pre result, an explicit state of the node
	  Value s = nodeOp.addState(preOp.getResult().getType());
	  preOp.getResult().replaceAllUsesWith(s);

	  // yield the explicit state at cycle k > 0 (the pre operand)
	  YieldOp formerYield = nodeOp.getYield();
	  builder.setInsertionPointAfter(formerYield);
	  OperationState state(loc, YieldOp::getOperationName());
	  SmallVector<Value, 4> newStates;
	  newStates.append(formerYield.getStates().begin(),
			   formerYield.getStates().end());
	  newStates.push_back(preOp.getOperand());
	  SmallVector<Value, 4> outputs;
	  outputs.append(formerYield.getOutputs().begin(),
			 formerYield.getOutputs().end());
	  YieldOp::build(builder, state, newStates, outputs);
	  Operation* yieldOpPtr = builder.createOperation(state);
	  formerYield.getOperation()->moveBefore(yieldOpPtr);
	
	  former.push_back(formerYield.getOperation());
	}

	// If dominance is granted in the node :
	// just clock the operand (absent on first cicle, present then)
	else {
	  vector<bool> nextPrefix;
	  nextPrefix.push_back(false);
	  vector<bool> nextPeriod;
	  nextPeriod.push_back(true);
	  KPeriodic nextKPeriodic(nextPrefix, nextPeriod);
	  Cond<Value> nextCond(nextKPeriodic);
	  OperationState nextState(loc, WhenOp::getOperationName());
	  WhenOp::build(builder, nextState, nextCond, preOp.getOperand());
	  Operation* whenOpPtr = builder.createOperation(nextState);
	  op->replaceAllUsesWith(whenOpPtr);
	}
	
	former.push_back(op);
      }
    }
  }
}
    
