#include "../../Dialects/Pssa/CreatePredOp.h"
#include "NodeToNode.h"
// #include "mlir/IR/Module.h"
#include "../../Dialects/Sync/InputOp.h"
#include "../../Dialects/Sync/OutputOp.h"
#include "../../Dialects/Sync/UndefOp.h"
#include "../../Dialects/Sync/TickOp.h"
#include "../../Dialects/Sync/HaltOp.h"
#include "../../Dialects/Sync/SyncOp.h"
#include "../../Dialects/Lus/Yield.h"
#include "..//Utilities/OperationsAux.h"
#include <list>

namespace mlir {
  namespace sync {

    void NodeToNode::operator() (ModuleOp moduleOp) {

      std::list<Operation*> topLevel;
      for (Operation &op : *(moduleOp.getBodyRegion().begin())) {
      	topLevel.push_back(&op);
      }
      for (Operation *op : topLevel) {
	if (isa<LusNodeOp>(op)) {
	  LusNodeOp lusNodeOp = dyn_cast<LusNodeOp>(op);
	  lower(lusNodeOp);
	}
      }
      for (Operation *op: dishes) {
	op->erase();
      }
      for (Block *b: dishesBlock) {
	b->erase();
      }
    }

    sync::NodeOp NodeToNode::lower(LusNodeOp lusNodeOp) {
      
      // Build the node skeleton
      SyncNodeOp syncNodeOp = buildSkeleton(lusNodeOp);
      Block *mainBlock = &lusNodeOp.getBody().front();
      ConstantPool constantPool(syncNodeOp);

      // Manage k-periodic words
      Operation *lusTermOp = mainBlock->getTerminator();
      OpBuilder updateBuilder(lusTermOp);
      std::list<Value> inits;
      std::list<Value> updates;
      std::list<Operation*> operations;
      for (Operation &innerOp : mainBlock->getOperations()) {
	operations.push_back(&innerOp);
      }
      for (Operation* innerOp: operations) {
	prepareKperiodicWord(constantPool, updateBuilder, mainBlock,
			     inits, updates, innerOp);
      }

      lowerInputs(syncNodeOp, lusNodeOp);

      // Pack the loop carried dependencies
      SmallVector<Value, 4> carriedValues;
      SmallVector<Type, 4> carriedTypes;
      for (Type ty : lusNodeOp.getStatesTypes()) {
	carriedTypes.push_back(ty);
      	Value v = constantPool.getUndef(ty);
      	carriedValues.push_back(v);
      }
      for (Value v: inits) {
	carriedValues.push_back(v);
	carriedTypes.push_back(v.getType());
      }

      OpBuilder builder = OpBuilder::atBlockEnd(&syncNodeOp.getBody()
      						  .front());
      // Version with scf.for
      Value z = constantPool.getZero(builder.getIndexType());
      Value o = constantPool.getOne(builder.getIndexType());
      Value m = constantPool.buildInt(builder.getIndexType(),
				      0x7fffffffffffffff);
      OperationState forState(syncNodeOp.getLoc(),
			      scf::ForOp::getOperationName());
      scf::ForOp::build(builder, forState, z, m, o, carriedValues);
      Operation *forOpPtr = builder.createOperation(forState);
      scf::ForOp forOp = dyn_cast<scf::ForOp>(forOpPtr);
      Block *loopBlock = &forOp.getLoopBody().front();
      // place for the induction variable
      mainBlock->insertArgument((unsigned)0, builder.getIndexType());
      
      // Version with scf.while
      // Value trueVal = constantPool.getBool(true);
      // scf::WhileOp whileOp = builder.create<scf::WhileOp>
      // 	(lusNodeOp.getLoc(),
      // 	 carriedTypes,
      // 	 carriedValues);
      // Block *beforeBlock = builder.createBlock(&whileOp.before(), {},
      // 					       carriedTypes);
      // builder.create<scf::ConditionOp>(lusNodeOp.getLoc(),
      // 				       trueVal,
      // 				       whileOp.before().getArguments());
      // Block *loopBlock = builder.createBlock(&whileOp.after());
      
      mainBlock->moveBefore(loopBlock);
      dishesBlock.push_back(loopBlock);

      // lower outputs
      lus::YieldOp yieldOp = dyn_cast<lus::YieldOp>(lusTermOp);
      builder.setInsertionPoint(yieldOp);
      // SmallVector<Value,4> outputSyncs;
      // for (auto e: llvm::zip(syncNodeOp.getOutputs(), yieldOp.getOutputs())) {
      // 	OutputOp o = builder.create<OutputOp>(yieldOp.getLoc(),
      // 					      std::get<0>(e), std::get<1>(e));
      // 	outputSyncs.push_back(o.getResult());
      // }
      // TickOp tickOp = builder.create<TickOp>(yieldOp.getLoc(), outputSyncs);
      // TickOp tickOp = builder.create<TickOp>(yieldOp.getLoc());

      // lower yield
      SmallVector<Value, 4> yieldVals;
      yieldVals.append(yieldOp.getStates().begin(),
      		       yieldOp.getStates().end());

      // SyncOp syncOp = builder.create<SyncOp>(yieldOp.getLoc(),
      // 					     tickOp.getResult(), yieldVals);
      // yieldVals.clear();
      // yieldVals.append(syncOp.getResults().begin(),
      // 		       syncOp.getResults().end());
      yieldVals.append(updates.begin(), updates.end());

      
      builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldVals);
      dishes.push_back(yieldOp);

      // Terminate the node
      builder.setInsertionPointToEnd(&syncNodeOp.getBody().front());
      OperationState endState(syncNodeOp.getLoc(),
    			      HaltOp::getOperationName());
      HaltOp::build(builder, endState);
      builder.createOperation(endState);
      
      dishes.push_back(lusNodeOp);

      return syncNodeOp;
    }
    
    void NodeToNode::lowerInputs(SyncNodeOp syncNodeOp, LusNodeOp lusNodeOp) {
      Block *nodeBlock = &lusNodeOp.getBody().front();
      OpBuilder builder = OpBuilder::atBlockBegin(nodeBlock);
      for (Value syncParam : syncNodeOp.getInputs()) {
	Value lusParam = nodeBlock->getArgument(0);
	OperationState inputState(lusNodeOp.getLoc(),
				  InputOp::getOperationName());
	InputOp::build(builder, inputState, syncParam);
	Operation *inputOpPtr = builder.createOperation(inputState);
	InputOp inputOp = dyn_cast<InputOp>(inputOpPtr);
	lusParam.replaceAllUsesWith(inputOp.getResult());
	nodeBlock->eraseArgument(0);
      }
    }
    
    sync::NodeOp NodeToNode::buildSkeleton(LusNodeOp lusNodeOp) {
      OpBuilder builder(lusNodeOp);
      OperationState nodeState(lusNodeOp.getLoc(),
			       SyncNodeOp::getOperationName());
      SyncNodeOp::build(builder, nodeState,
			lusNodeOp.getNodeName(),
			lusNodeOp.getStaticTypes(),
			lusNodeOp.getInputsTypes(),
			lusNodeOp.getOutputsTypes(),
			lusNodeOp.hasBody(),
			lusNodeOp.getStatesTypes());
      Operation *syncNodeOpPtr = builder.createOperation(nodeState);
      SyncNodeOp syncNodeOp = dyn_cast<SyncNodeOp>(syncNodeOpPtr);
      return syncNodeOp;
    }


    void NodeToNode::prepareKperiodicWord(ConstantPool &constantPool,
					  OpBuilder &updateBuilder,
					  Block *nodeBlock,
					  std::list<Value> &inits,
					  std::list<Value> &updates,
					  Operation *op) {
      for (Region &r: op->getRegions()) {
	for (Block& b: r.getBlocks()) {
	  for (Operation& deeperOp: b.getOperations()) {
	    prepareKperiodicWord(constantPool, updateBuilder, nodeBlock,
				 inits, updates, &deeperOp);
	  }
	}
      }
      if (auto createPredOp = dyn_cast<pssa::CreatePredOp>(op)) {
	if (createPredOp.isKPeriodic()) {
	  OpBuilder createPredOpBuilder(createPredOp);
	  KPeriodic kp = createPredOp.word();
	  Value predicate;
	  if (kp.isHeadTail()) {
	    bool prefix = kp.getPrefix().front();
	    bool period = kp.getPeriod().front();
	    Value headVal = constantPool.getBool(prefix);
	    Value tailVal = constantPool.getBool(period);
	    predicate = nodeBlock->addArgument(updateBuilder.getI1Type());
	    inits.push_back(headVal);
	    updates.push_back(tailVal);
	  }
	  else {
	    std::vector<bool> pattern;
	    pattern.insert(pattern.end(),
			   kp.getPrefix().begin(), kp.getPrefix().end());
	    pattern.insert(pattern.end(),
			   kp.getPeriod().begin(), kp.getPeriod().end());
	    Value vectorConst = constantPool.getVector(pattern,
						       updateBuilder.getI32Type());
	    Value initCounter = constantPool.getZero(updateBuilder.getI32Type());
	    inits.push_back(initCounter);
	    Value counter = nodeBlock->addArgument(updateBuilder.getI32Type());
	    predicate = constantPool.extract(createPredOpBuilder,
					     vectorConst, counter);
	    Value floor = constantPool.buildInt(updateBuilder.getI32Type(),
						kp.getPrefix().size());
	    Value ceiling = constantPool.buildInt(updateBuilder.getI32Type(),
						  kp.getPeriod().size());
	    Value incr = constantPool.increment(updateBuilder, counter);
	    CmpIOp cmpiOp = updateBuilder.create<CmpIOp>(createPredOp.getLoc(),
							 CmpIPredicate::sge,
							 incr, ceiling);
	    OperationState selectState(createPredOp.getLoc(),
				       SelectOp::getOperationName());
	    SelectOp::build(updateBuilder, selectState,
			    cmpiOp.getResult(), floor, incr);
	    Operation *selectOpPtr = updateBuilder.createOperation(selectState);
	    SelectOp selectOp = dyn_cast<SelectOp>(selectOpPtr);
	    updates.push_back(selectOp.getResult());
	  }

	  OperationState predState(op->getLoc(),
				   pssa::CreatePredOp::getOperationName());
	  pssa::CreatePredOp::build(&createPredOpBuilder, predState, predicate);
	  Operation *newCreatePredOpPtr = updateBuilder.createOperation(predState);
	  createPredOp.replaceAllUsesWith(newCreatePredOpPtr);
	  dishes.push_back(createPredOp);
	}
      }
    }
  }
}
