#include "LowerPssa.h"
#include "../Utilities/OperationsAux.h"
#include "../../Dialects/Sync/UndefOp.h"
#include "../../Dialects/Sync/OutputOp.h"
#include "mlir/Dialect/SCF/SCF.h"
#include <string>

namespace mlir {
  namespace pssa {

    void LowerPssa::operator() (sync::NodeOp nodeOp) {

      constantPool = new ConstantPool(nodeOp);

      lower(nodeOp, nodeOp);

      for (Operation *op: dishes) {
      	op->erase();
      }
      
      delete constantPool;
    }

    void LowerPssa::lower(sync::NodeOp nodeOp, Operation *op) {

      // apply on nested operations
      for (Region &r: op->getRegions()) {
	for (Operation &deeperOp: *r.begin()) {
	  lower(nodeOp, &deeperOp);
	}
      }

      // pattern matching
      if (isa<CreatePredOp>(op)) {
	CreatePredOp createPredOp = dyn_cast<CreatePredOp>(op);
	lowerCreatePredOp(createPredOp);
      }
      else if (isa<CondactOp>(op)) {
	CondactOp condactOp = dyn_cast<CondactOp>(op);
	lowerCondactOp(condactOp);
      }
      else if (isa<YieldOp>(op)) {
	YieldOp yieldOp = dyn_cast<YieldOp>(op);
	lowerYieldOp(yieldOp);
      }
      else if (isa<OutputOp>(op)) {
	OutputOp outputOp = dyn_cast<OutputOp>(op);
	lowerOutputOp(nodeOp, outputOp);
      }
    }

    void LowerPssa::lowerOutputOp(NodeOp nodeOp, OutputOp outputOp) {
      Location loc = outputOp.getLoc();
      int64_t pos = outputOp.getPosition();
      Value sig = nodeOp.getOutput(pos);
      Value out = outputOp.getOperand();
      OpBuilder builder(outputOp);
      sync::OutputOp syncOutputOp = builder.create<sync::OutputOp>(loc,
								   sig,
								   out);
      outputOp.getResult().replaceAllUsesWith(syncOutputOp.getResult());
      dishes.push_back(outputOp);
    }

    void LowerPssa::lowerCreatePredOp(CreatePredOp createPredOp) {
      OpBuilder builder(createPredOp);
      Value truePred = createPredOp.getResult(0);
      Value falsePred = createPredOp.getResult(1);
      
      // If the complement has uses, generate it
      if (!falsePred.use_empty()) {
	Value neg = constantPool->negate(builder, createPredOp.data());
	createPredOp.getResult(1).replaceAllUsesWith(neg);
      }

      // Use the initial boolean
      truePred.replaceAllUsesWith(createPredOp.data());
      
      dishes.push_back(createPredOp);
    }

    void LowerPssa::lowerCondactOp(CondactOp condactOp) {
      
      // Condact doesn't contain operations (except yield)
      if (isa<YieldOp>(condactOp.getBody().front().front())) {
	auto ops = condactOp.getBody().front().getTerminator()->getOperands();
	condactOp.replaceAllUsesWith(ops);
      }

      else {

	// Get default results of the condact
	SmallVector<Value, 4> myDefaults;
	myDefaults.append(condactOp.defaults().begin(),
			  condactOp.defaults().end());
	if (myDefaults.empty()) {
	  for (Value r: condactOp.getResults()) {
	    Value v = constantPool->getUndef(r.getType());
	    myDefaults.push_back(v);
	  }
	}

	OpBuilder builder(condactOp);
	OperationState ifOpState(condactOp.getLoc(),
				 scf::IfOp::getOperationName());
	scf::IfOp::build(builder, ifOpState,
			 condactOp.getResultTypes(), condactOp.condition(),
			 true);
	Operation *ifOpPtr = builder.createOperation(ifOpState);
	scf::IfOp ifOp = dyn_cast<scf::IfOp>(ifOpPtr);
	Region &thenRegion = ifOp.thenRegion();
	thenRegion.takeBody(condactOp.getBody());
	Region &elseRegion = ifOp.elseRegion();
	builder.setInsertionPoint(&elseRegion.front(),
				  elseRegion.front().begin());
	builder.create<scf::YieldOp>(condactOp.getLoc(),
				     myDefaults);

	condactOp.replaceAllUsesWith(ifOpPtr);
      }
      
      dishes.push_back(condactOp);
    }

    void LowerPssa::lowerYieldOp(YieldOp yieldOp) {
      OpBuilder builder(yieldOp);
      OperationState yieldState(yieldOp.getLoc(),
				scf::YieldOp::getOperationName());
      scf::YieldOp::build(builder, yieldState, yieldOp.getOperands());
      builder.createOperation(yieldState);
      dishes.push_back(yieldOp);
    }
  }
}
