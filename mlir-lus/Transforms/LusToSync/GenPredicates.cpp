#include "GenPredicates.h"
#include "../../Dialects/Pssa/CreatePredOp.h"
#include "../../Dialects/Lus/lus.h"

using namespace mlir::pssa;

namespace mlir {
  namespace lus {

    void GenPredicates::operator() (NodeOp nOp) {

      nodeOp = nOp;

      // Handle each when, merge or pred operations
      for (Operation& op : *(nodeOp.getBody().begin())) {
	if (isa<WhenOp>(&op)) {
	  WhenOp whenOp = dyn_cast<WhenOp>(&op);
	  handle(whenOp);
	}
	else if (isa<MergeOp>(op)) {
	  MergeOp mergeOp = dyn_cast<MergeOp>(&op);
	  handle(mergeOp);
	}
	else if (isa<CreatePredOp>(op)) {
	  CreatePredOp createPredOp = dyn_cast<CreatePredOp>(&op);
	  handle(createPredOp);
	}
      }
      
      finalize();
    }

    void GenPredicates::handle(CreatePredOp createPredOp) {
      if (createPredOp.isDataDependent()) {
	result.set(createPredOp.data(), createPredOp);
      }
      else if (createPredOp.isKPeriodic()) {
	result.set(createPredOp.word(), createPredOp);
      }
    }
    
    void GenPredicates::handle(WhenOp whenOp) {
      if (whenOp.getCondType().getType() == CondDataType) {
	Value v = whenOp.getCondValue();
	buildCreatePredOp(v);
      }
      else if (whenOp.getCondType().getType() == CondKPType) {
	KPeriodic kp = whenOp.getCondKPeriodic();
	buildCreatePredOp(kp);
      }
    }

    void GenPredicates::handle(MergeOp mergeOp) {
      if (mergeOp.getCondType().getType() == CondDataType) {
	Value v = mergeOp.getCondValue();
	buildCreatePredOp(v);
      }
      else if (mergeOp.getCondType().getType() == CondKPType) {
	KPeriodic kp0 = mergeOp.getCondKPeriodic();
	KPeriodic kp1(kp0.getPrefix(), kp0.getPeriod());
	KPeriodic kp2 = kp1.buildComplement();
	buildCreatePredOp(kp1);
	buildCreatePredOp(kp2);
      }
    }

    void GenPredicates::finalize() {
      // The predicate corresponding to state values' clock
      if (nodeOp.getNumStates() > 0) {
      	vector<bool> prefix;
	prefix.push_back(false);
	vector<bool> period;
	period.push_back(true);
      	KPeriodic kp(prefix,period);
      	buildCreatePredOp(kp);
      }
    }
    
    void GenPredicates::buildCreatePredOp(Value condition) {
      if (result.count(condition) == 0) {
	OpBuilder builder(&nodeOp.getBody());
	// If condition has a defining op (then it's not a node parameter),
	// write predicate after it.
	if (auto defOp = condition.getDefiningOp()) {
	  builder.setInsertionPointAfter(defOp);
	}
	OperationState state(condition.getLoc(),
			     CreatePredOp::getOperationName());
	CreatePredOp::build(&builder, state, condition);
	Operation *op = builder.createOperation(state);
	result.set(condition, op);
      }
    }

    void GenPredicates::buildCreatePredOp(KPeriodic condition) {
      if (result.count(condition) == 0) {
	OpBuilder builder(&nodeOp.getBody());
	OperationState state(nodeOp.getLoc(),
			     CreatePredOp::getOperationName());
	CreatePredOp::build(&builder, state, condition);
	auto op = builder.createOperation(state);
	result.set(condition, op);
      }
    }

    
    
    

    
    
  }
}
