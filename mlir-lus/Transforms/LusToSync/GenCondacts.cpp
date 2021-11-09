#include "GenCondacts.h"
#include "../../Dialects/Pssa/CondactOp.h"
#include "../../Dialects/Pssa/CreatePredOp.h"
#include "../../Dialects/Lus/ClockAnalysis.h"
#include "../../Dialects/Lus/WhenOp.h"
#include "../../Dialects/Lus/MergeOp.h"
#include "../../Dialects/Sync/SelectOp.h"
#include "../../Dialects/Sync/TickOp.h"
#include "../../Dialects/Sync/SyncOp.h"
#include <algorithm> 

using namespace mlir::pssa;

namespace mlir {
  namespace lus {

    void GenCondacts::operator() (NodeOp nodeOp) {

      // Parse predicates in the node
      for (Operation& op : *(nodeOp.getBody().begin())) {
	storePredicates(&op);
      }

      // Analyse clocks
      ClockAnalysis analysis(nodeOp);
      assert(succeeded(analysis.analyse()));
      clockTree = &analysis.getClockTree();

      // Gen condacts (using clocks and predicates)
      list<Operation*> buffer;
      for (Operation& op : *(nodeOp.getBody().begin())) {
      	if (shouldBeCondacted(&op)) {
      	  buffer.push_back(&op);
      	}
      }
      for (Operation *op: buffer) {
	Operation* newOp = insertInCondacts(op);
	updateEnvironment(op, newOp);
      }

      // Lowering of merge and when
      for (Operation& op : *(nodeOp.getBody().begin())) {
      	removeReactive(&op);
      }

      for (Operation* op : dishes) {
      	op->erase();
      }
    }

    void GenCondacts::storePredicates(Operation *op) {

      if (isa<CreatePredOp>(op)) {
	CreatePredOp createPredOp = dyn_cast<CreatePredOp>(op);
	if (createPredOp.isDataDependent()) {
	  condToPred.set(createPredOp.data(), op);
	}
	else if (createPredOp.isKPeriodic()) {
	  condToPred.set(createPredOp.word(), op);
	}
      }
    }

    bool GenCondacts::shouldBeCondacted(Operation *op) {
      // If op is yield or is clocked on system clock : nothing to do
      if (isa<YieldOp>(op) || isa<sync::SyncOp>(op) || isa<sync::TickOp>(op)
	  || clockTree->path(op).empty())
	return false;
      return true;
    }

    Operation* GenCondacts::insertInCondacts(Operation* op) {
      OpBuilder builder = OpBuilder(op);
      
      auto risingOp = op;

      for (ClockTree::Edge edge : clockTree->path(op)) {

	// Build the condact
	builder.setInsertionPointAfter(risingOp);
	OperationState condactState(builder.getUnknownLoc(),
				    CondactOp::getOperationName());
	SmallVector<Type,4> condactTypes;
	condactTypes.append(risingOp->getResultTypes().begin(),
			    risingOp->getResultTypes().end());
	CondactOp::build(builder, condactState,
			 condToPred.getValue(edge),
			 condactTypes);
			 // risingOp->getResultTypes(),
	Operation* predicatedOp = builder.createOperation(condactState);
	CondactOp condactOp = dyn_cast<CondactOp>(predicatedOp);
	risingOp->replaceAllUsesWith(predicatedOp);
	// Fill the condact
	pssa::YieldOp yieldOp = condactOp.getYield();
	risingOp->moveBefore(yieldOp);
	yieldOp->setOperands(risingOp->getResults());

	risingOp = predicatedOp;
      }
      
      return risingOp;
    }
    
    void GenCondacts::updateEnvironment(Operation* op, Operation *newOp) {

      if (isa<CreatePredOp>(op)) {

	// If you had cond -> pred and you do pred' = condact { pred },
	// you now have cond -> pred'
	CreatePredOp createPredOp = dyn_cast<CreatePredOp>(op);
	if (createPredOp.isDataDependent()) {
	  condToPred.set(createPredOp.data(), newOp);
	}
	else if (createPredOp.isKPeriodic()) {
	  condToPred.set(createPredOp.word(), newOp);
	}
      }

      // If you had v = op and v was a condition, you need to update v as
      // clockTree conditions (edges) and as condToPred keys
      for (auto e : llvm::zip(op->getResults(), newOp->getResults())) {
      	Value oldValue = get<0>(e);
      	Value newValue = get<1>(e);
      	if (condToPred.count(oldValue) > 0) {
      	  clockTree->substitute(oldValue, newValue);
      	  condToPred.set(newValue, condToPred.get(oldValue));
      	}
      }
    }
    
    void GenCondacts::removeReactive(Operation* op) {

      // Apply in nested regions
      for (Region& region : op->getRegions()) {
	for (Operation &deeperOp : *(region.begin())) {
	  removeReactive(&deeperOp);
	}
      }

      if (isa<WhenOp>(op)) {
	// Just replace when by its operand
	auto whenOp = dyn_cast<WhenOp>(op);
	whenOp.getResult().replaceAllUsesWith(whenOp.getDataInput());
	dishes.insert(op);
      }
      
      else if (isa<MergeOp>(op)) {
	
	MergeOp mergeOp = dyn_cast<MergeOp>(op);
	
	// Get predicate corresponding to merge condition
	Operation* predOpPtr;
	if (mergeOp.isCondData()) {
	  predOpPtr = condToPred.get(mergeOp.getCondValue());
	}
	else if (mergeOp.isCondKPeriodic()) {
	  predOpPtr = condToPred.get(mergeOp.getCondKPeriodic());
	}
	else {
	  assert(false);
	}

	// Replace merge by select
	OpBuilder builder(op);
	OperationState state(op->getLoc(),sync::SelectOp::getOperationName());
	sync::SelectOp::build(builder, state,
			      predOpPtr->getResult(0), // true predicate
			      mergeOp.getTrueInput(),mergeOp.getFalseInput());
	Operation * selectOpPtr = builder.createOperation(state);
	sync::SelectOp selectOp = dyn_cast<sync::SelectOp>(selectOpPtr);
	mergeOp.replaceAllUsesWith(selectOpPtr);
	updateEnvironment(mergeOp, selectOp);

	dishes.insert(op);
      }
    }

    
    
    
    
  }
}
