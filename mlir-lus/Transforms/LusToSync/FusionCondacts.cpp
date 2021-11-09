#include "mlir/Pass/Pass.h" // For ModuleOp
#include "FusionCondacts.h"
#include "../../Dialects/Lus/Node.h"
// #include "mlir/IR/Function.h"
#include "mlir/IR/AsmState.h"

namespace mlir {
  namespace pssa {

    void FusionCondacts::operator()(lus::NodeOp nodeOp) {
      apply(nodeOp.getBody());
    }
    
    void FusionCondacts::apply(Region &body) {

      // Acts on (ordered) ops of the body ; for each op n + 1, try to fusion
      // as a condact with the op n
      list<Operation*> ops;
      for (Operation &op : *(body.begin())) {
	ops.push_back(&op);
      }
      CondactOp prev;
      bool prevIsCondact = false;
      
      for (Operation *op : ops) {

	// Current and previous ops are condacts, and they can fusion :
	// - fusion
	// - the fusionned result is now the previous op.
	// - prevIsCondact unchanged (true) : the new previous op is the
	//   fusionned condact
	if (isa<CondactOp>(op)
	    && prevIsCondact
	    && canFusion(prev, dyn_cast<CondactOp>(op))) {
	  CondactOp currentCondact = dyn_cast<CondactOp>(op);
	  prev = fusion(prev, currentCondact);
	}

	// Current and previous ops are condacts, but they can't fusion :
	// - apply on the previous op
	// - the current op is now the previous op.
	// - now remember that the previous op is a condact
	else if (isa<CondactOp>(op) && prevIsCondact) {
	  apply(prev.getBody());
	  prev = dyn_cast<CondactOp>(op);
	  prevIsCondact = true;
	}

	// Current op is condact but not previous op :
	// - the current op is now the previous op.
	// - now remember that the previous op is a condact
	else if (isa<CondactOp>(op) && !prevIsCondact) {
	  prev = dyn_cast<CondactOp>(op);
	  prevIsCondact = true;
	}

	// Current op is not condact :
	// - apply on the previous op
	// - now remember that the previous op isn't a condact
	else if (prevIsCondact) {
	  apply(prev.getBody());
	  prevIsCondact = false;
	}
	// Current and previous ops are not condacts : skip
	else if (!prevIsCondact) { }
      }
    }

    bool FusionCondacts::canFusion(CondactOp prev, CondactOp curr) {
      // can fusion iff share the same condition
      return prev.condition() == curr.condition();
    }
    
    CondactOp FusionCondacts::fusion(CondactOp fst, CondactOp snd) {
      assert(canFusion(fst, snd));
      
      list<Operation*> dishes;
      OpBuilder builder(fst);
      
      builder.setInsertionPointAfter(snd);

      // Build an empty condact whith the right outputs types
      SmallVector<Type, 4> resultTypes;
      resultTypes.append(fst.getResults().getTypes().begin(),
			 fst.getResults().getTypes().end());
      resultTypes.append(snd.getResults().getTypes().begin(),
			 snd.getResults().getTypes().end());
      OperationState condactState(builder.getUnknownLoc(),
				  CondactOp::getOperationName());
      CondactOp::build(builder, condactState, fst.condition(), resultTypes);
      Operation* condactOpPtr = builder.createOperation(condactState);
      CondactOp condactOp = dyn_cast<CondactOp>(condactOpPtr);

      // Build the new yield (inside the empty condact)
      YieldOp initialYield = condactOp.getYield();
      builder.setInsertionPointAfter(initialYield);
      SmallVector<Value, 4> yieldFusion;
      yieldFusion.append(fst.getYield().getOperands().begin(),
			 fst.getYield().getOperands().end());
      yieldFusion.append(snd.getYield().getOperands().begin(),
			 snd.getYield().getOperands().end());
      OperationState yieldState(builder.getUnknownLoc(),
				YieldOp::getOperationName());
      YieldOp::build(builder, yieldState, yieldFusion);
      Operation *yieldOpPtr = builder.createOperation(yieldState);
      YieldOp yieldOp = dyn_cast<YieldOp>(yieldOpPtr);
      dishes.push_back(initialYield);

      // Fill the new condact
      builder.setInsertionPointAfter(yieldOp);
      builder.clone(*yieldOp.getOperation());
      dishes.push_back(yieldOp);
      list<Operation*> body;
      for (Operation &op : *(fst.getBody().begin())) {
      	if (!isa<YieldOp>(&op)) {
      	  body.push_back(&op);
      	}
      }
      for (Operation &op : *(snd.getBody().begin())) {
      	if (!isa<YieldOp>(&op)) {
      	  body.push_back(&op);
      	}
      }
      Operation* prevOp = yieldOp.getOperation();
      for (Operation* op: body) {
      	op->moveAfter(prevOp);
      	prevOp = op;
      }

      // Replace values
      SmallVector<Value, 4> resultsFusion;
      resultsFusion.append(fst.getResults().begin(), fst.getResults().end());
      resultsFusion.append(snd.getResults().begin(), snd.getResults().end());
      replaceResultsOutside(resultsFusion, condactOp);
      replaceResultsInside(fst, condactOp);

      // Clean
      dishes.push_back(fst);
      dishes.push_back(snd);
      for (Operation* op : dishes) {
      	op->erase();
      }

      return condactOp;
    }

    void FusionCondacts::replaceResultsOutside(ArrayRef<Value> resultsFusion,
					       CondactOp condactOp) {
      auto myIf = [&](OpOperand &operand) {
		    Operation *owner = operand.getOwner();
		    if (!isa<lus::NodeOp>(owner->getParentOp()))
		      while (!isa<lus::NodeOp>(owner) &&
			     !isa<FuncOp>(owner)) {
			if (owner == condactOp.getOperation()) {
			  return false;
			}
			owner = owner->getParentOp();
		      }
		    return true;
		  };
      for (auto e : llvm::zip(resultsFusion, condactOp.getResults())) {
      	Value old = get<0>(e);
      	Value neo = get<1>(e);
      	old.replaceUsesWithIf(neo, myIf);
      }
    }

    void FusionCondacts::replaceResultsInside(CondactOp fst,
					      CondactOp condactOp) {

      auto myIf = [&](OpOperand &operand) {
		    Operation *owner = operand.getOwner();
		    if (!isa<lus::NodeOp>(owner->getParentOp()))
		      while (!isa<lus::NodeOp>(owner) &&
			     !isa<FuncOp>(owner)) {
			if (owner == condactOp.getOperation()) {
			  return true;
			}
			owner = owner->getParentOp();
		      }
		    return false;
		  };
      
      for (auto e : llvm::zip(fst.getResults(),
			      fst.getYield().getOperands())) {
	get<0>(e).replaceUsesWithIf(get<1>(e), myIf);
      }
    }
  }
}
