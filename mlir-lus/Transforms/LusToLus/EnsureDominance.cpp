#include "EnsureDominance.h"
#include "../../Dialects/Lus/ClockAnalysis.h"

namespace mlir {
  namespace lus {

    void EnsureDominance::operator() (NodeOp nodeOp) {

      // Dominance is ensured yed
      if (!nodeOp.isDominanceFree())
	return;
      
      // Clock analysis
      ClockAnalysis analysis(nodeOp);
      analysis.analyse();
      clockTree = &analysis.getClockTree();

      // input and state variables are always available
      for (Value v : nodeOp.getBody().getArguments()) {
	definedValues.insert(v);
      }
      
      // The operations which must be sorted (all except yield)
      for (Operation& op : *(nodeOp.getBody().begin())) {
	if (!isa<YieldOp>(&op)) {
	  unorderedOps.push_back(&op);
	}
      }

      // sort unordered ops
      order();
      assert(unorderedOps.size() == 0);
      // add yield at the end of the sorted list
      orderedOps.push_back(nodeOp.getYield().getOperation());

      // write the ordered list into the node
      orderedOps.front()->moveBefore(&nodeOp.getBody().front().front());
      Operation* prev = orderedOps.front();
      orderedOps.pop_front();
      for (Operation* op : orderedOps) {
	op->moveAfter(prev);
	prev = op;
      }

      // Change dominance flag
      nodeOp.forceDominance();
    }

    void EnsureDominance::order() {

      // Initialization : first operation is tiniest clock among unordered
      // ops
      bool onAny = nextOnTiniestClock();
      bool onSub = false;
      bool onSame = false;

      // This loop ends when no operation can be sorted
      while(onAny || onSub || onSame) {
	// Try to sort an operation which is on the same clock
    	onSame = nextOnSameClock();
    	if (!onSame) {
	  // Try to sort an operation which is on a sub clock
    	  onSub = nextOnSubClock();
    	}
    	if (!onSub && !onSame) {
	  // Try to sort an operation which is on an other clock (the tiniest
	  // among unordered ops)
    	  onAny = nextOnTiniestClock();
    	}
      } 
    }
    
    bool EnsureDominance::dominanceCorrection(Operation* op) {
      
      // Check if all operands of op are already defined
      for (Value v : op->getOperands()) {
	if (definedValues.count(v) == 0) {
	  return false;
	}
      }

      // Check if all clocks of op are already defined
      for (ClockTree::Edge e: clockTree->path(op)) {
	if (e.getType() == CondDataType) {
	  if (definedValues.count(e.getData()) == 0) {
	    return false;
	  }
	}
      }
      return true;
    }

    bool EnsureDominance::nextOnSameClock() {
      Operation *lastOp = orderedOps.back();
      list<Operation*>::iterator it;
      it = unorderedOps.begin();
      while (it != unorderedOps.end()) {
	Operation* op = *it;
	if (dominanceCorrection(op)
	    && clockTree->equal(lastOp, op)) {
	  for (Value v : op->getResults()) {
	    definedValues.insert(v);
	  }
	  orderedOps.push_back(op);
	  it = unorderedOps.erase(it);
	  return true;
	}
	else {
	  it++;
	}
      }
      return false;
    }
    
    bool EnsureDominance::nextOnSubClock() {
      Operation *lastOp = orderedOps.back();
      list<Operation*>::iterator it;
      it = unorderedOps.begin();
      while (it != unorderedOps.end()) {
	Operation* op = *it;
	if (dominanceCorrection(op)
	    && clockTree->greater(lastOp, op)) {
	  for (Value v : op->getResults()) {
	    definedValues.insert(v);
	  }
	  orderedOps.push_back(op);
	  it = unorderedOps.erase(it);
	  return true;
	}
	else {
	  it++;
	}
      }
      return false;
    }

    bool EnsureDominance::nextOnTiniestClock() {
      Operation *tiniest;
      int tiniestSize;
      bool zeroFound = true;
      unsigned i = 0;
      for (Operation *op : unorderedOps) {
	ClockTree::Path p = clockTree->path(op);
	if ((zeroFound || p.size() < tiniestSize)
	    && dominanceCorrection(op)) {
	  tiniest = op;
	  tiniestSize = p.size();
	  zeroFound = false;
	}
	i++;
      }
      if (!zeroFound) {
	for (Value v : tiniest->getResults()) {
	  definedValues.insert(v);
	}
	orderedOps.push_back(tiniest);
	unorderedOps.remove(tiniest);
      }
      return !zeroFound;
    }
    
    

    
    
  }
}
