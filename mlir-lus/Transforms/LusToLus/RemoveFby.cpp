#include "RemoveFby.h"
#include <vector>
#include "../../Dialects/Lus/FbyOp.h"
#include "../../Dialects/Lus/PreOp.h"
#include "../../Dialects/Lus/WhenOp.h"
#include "../../Dialects/Lus/MergeOp.h"

namespace mlir {
  namespace lus {

    void RemoveFby::operator() (NodeOp nop) {
      for (Operation& op : *(nop.getBody().begin())) {
	removeFby(&op);
      }
      for (Operation* op : former) {
	op->erase();
      }
    }

    void RemoveFby::removeFby(Operation* op) {

      // apply in nested regions
      for (Region& r : op->getRegions()) {
	for (Operation& innerOp : *(r.begin())) {
	  removeFby(&innerOp);
	}
      }
      
      if (isa<FbyOp>(op)) {

	FbyOp fbyOp = dyn_cast<FbyOp>(op);
	Location loc = op->getLoc();
	OpBuilder builder = OpBuilder(op);

	// Clock the initial value (present on first cycle, absent then)
	vector<bool> initPrefix;
	initPrefix.push_back(true);
	vector<bool> initPeriod;
	initPeriod.push_back(false);
	KPeriodic initKPeriodic(initPrefix, initPeriod);
	Cond<Value> initCond(initKPeriodic);
	OperationState initState(loc, WhenOp::getOperationName());
	WhenOp::build(builder, initState, initCond, fbyOp.getLhs());
	Operation* initOp = builder.createOperation(initState);
	WhenOp initWhen = dyn_cast<WhenOp>(initOp);

	// Represent the next value as a pre
	Value preResult;
	OperationState preState(loc, PreOp::getOperationName());
	PreOp::build(builder, preState, fbyOp.getRhs());
	Operation* preOpPtr = builder.createOperation(preState);
	PreOp preOp = dyn_cast<PreOp>(preOpPtr);
	preResult = preOp.getResult();

	// Merge initial and next value : it is the value of the fby op
	Cond<Value> mergeCond(initCond);
	OperationState mergeState(loc, MergeOp::getOperationName());
	MergeOp::build(builder, mergeState,
		       mergeCond, initWhen.getResult(), preResult);
	Operation* resOp = builder.createOperation(mergeState);
	MergeOp mergeOp = dyn_cast<MergeOp>(resOp);
	fbyOp.getResult().replaceAllUsesWith(mergeOp.getResult());
	
	former.push_back(op);
      }
    }
    
    

  }
}
