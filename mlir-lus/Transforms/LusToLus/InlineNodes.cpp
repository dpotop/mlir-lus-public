#include "InlineNodes.h"

namespace mlir {
  namespace lus {

    void InlineNodes::operator() (ModuleOp moduleOp) {
      packToDoNodes(moduleOp);
      while (!toDo.empty()) {
	packReadyNodes();
	inlineReadyNodes(moduleOp);
      }
    }

    void InlineNodes::packToDoNodes(ModuleOp moduleOp) {
      for (Operation &nop: *(moduleOp.getBodyRegion().begin())) {
	if (auto nodeOp = dyn_cast<NodeOp>(&nop)) {
	  if (nodeOp.hasBody()) {
	    for (Operation& op : *(nodeOp.getBody().begin())) {
	      if (auto instanceOp = dyn_cast<InstanceOp>(&op)) {
		NodeOp callee = instanceOp.getCalleeNode();
		toDo.insert(callee.getOperation());
	      }
	    }
	  }
	}
      }
    }

    bool InlineNodes::containsInstance(NodeOp nodeOp) {
      for (Operation &op: *(nodeOp.getBody().begin())) {
	if (isa<InstanceOp>(op)) {
	  return true;
	}
      }
      return false;
    }

    void InlineNodes::packReadyNodes() {
      unordered_set<Operation*>::iterator i = toDo.begin();
      while (i != toDo.end()) {
	NodeOp nodeOp = dyn_cast<NodeOp>(*i);
	if (!containsInstance(nodeOp)) {
	  ready.insert(nodeOp.getOperation());
	  toDo.erase(i++);
	}
	else {
	  i++;
	}
      }
    }

    void InlineNodes::inlineReadyNodes(ModuleOp moduleOp) {
      for (Operation &nop: *(moduleOp.getBodyRegion().begin())) {
	if (auto nodeOp = dyn_cast<NodeOp>(&nop)) {
	  for (Operation& op : *(nodeOp.getBody().begin())) {
	    if (auto instanceOp = dyn_cast<InstanceOp>(&op)) {
	      Operation *callee = instanceOp.getCalleeNode().getOperation();
	      if (ready.find(callee) != ready.end()) {
		inlineNode(nodeOp, instanceOp);
	      }
	    }
	  }
	  for (Operation *op: dishes) {
	    op->erase();
	  }
	  dishes.clear();
	}
      }
      done.insert(ready.begin(), ready.end());
      ready.clear();
    }

    void InlineNodes::inlineNode(NodeOp caller, InstanceOp instanceOp) {
      Operation *calleeClone = instanceOp
	.getCalleeNode()
	.getOperation()
	->clone();
      NodeOp calleeNode = dyn_cast<NodeOp>(calleeClone);
      Region& calleeBody = calleeNode.getBody();
      
      OpBuilder builder(instanceOp.getCalleeNode());
      builder.insert(calleeClone);

      // Set concrete parameters on a line
      SmallVector<Value, 4> concreteArgs;
      for (Value arg : instanceOp.getOperands()) {
	concreteArgs.push_back(arg);
      }
      for (Type stateType : calleeNode.getStatesTypes()) {
	// Update the caller signature
	Value s = caller.addState(stateType);
	concreteArgs.push_back(s);
      }
      // Set concrete parameters instead of abstract parameters
      for (auto e : llvm::zip(calleeBody.getArguments(), concreteArgs)) {
	Value absArg = get<0>(e);
	Value concArg = get<1>(e);
	absArg.replaceAllUsesWith(concArg);
      }

      // Move the callee operations inside the caller node
      list<Operation*> shouldBeInlined;
      for (Operation& calleeOp : *(calleeBody.begin())) {
	shouldBeInlined.push_back(&calleeOp);
      }
      Operation* currentOp = instanceOp.getOperation();
      for (Operation* calleeOp : shouldBeInlined) {
	if (!isa<YieldOp>(calleeOp)) {
	  calleeOp->moveAfter(currentOp);
	  currentOp = calleeOp;
	}
      }

      // Update the caller yield
      for (Value yieldState : calleeNode.getYield().getStates()) {
	caller.getYield().addState(yieldState);
      }

      // Use inlined results instead of instance
      for (auto e : llvm::zip(instanceOp.getResults(),
			      calleeNode.getYield().getOutputs())) {
	Value instanceRes = get<0>(e);
	Value yieldArg = get<1>(e);
	instanceRes.replaceAllUsesWith(yieldArg);
      }
	
      // Prepare the emptied callee to be erased
      dishes.insert(calleeNode.getOperation());
      dishes.insert(instanceOp.getOperation());
    }
  }
}
