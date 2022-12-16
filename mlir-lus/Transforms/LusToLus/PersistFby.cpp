#include "PersistFby.h"
#include <vector>
#include "../../Dialects/Lus/FbyOp.h"
#include "../../Dialects/Lus/PreOp.h"
#include "../../Dialects/Lus/WhenOp.h"
#include "../../Dialects/Lus/MergeOp.h"
#include "../../Dialects/Lus/ClockAnalysis.h"
#include "llvm/ADT/SmallPtrSet.h"
namespace mlir {
  namespace lus {

    void PersistFby::operator() (NodeOp nop) {
      nodeOp = nop.getOperation();
      bool fixedPoint;
      do {
	ClockAnalysis analysis(nop);
	analysis.analyse();
	clockTree = &analysis.getClockTree();
	fixedPoint = true;
	for (Operation& op : *(nop.getBody().begin())) {
	  ClockTree::Path p = clockTree->path(&op);
	  if (isa<FbyOp>(&op) && !p.empty()) {
	      persistFby(&op, p.back().getData());
	      fixedPoint = false;
	      break;
	    }
	  }
      } while(!fixedPoint);
      return;
    }

    void PersistFby::persistFby(Operation* op, Value valCond) {
      FbyOp fbyOp = dyn_cast<FbyOp>(op);
      Location loc = op->getLoc();
      OpBuilder builder = OpBuilder(op);
      Cond<Value> cond(valCond,false);
      Cond<Value> condNot(valCond,true);

      Value retFby = fbyOp.getResult();
      
      OperationState when1State(loc, WhenOp::getOperationName());
      WhenOp::build(builder, when1State, cond, retFby);
      Operation* when1OpPtr = builder.createOperation(when1State);
      WhenOp when1Op = dyn_cast<WhenOp>(when1OpPtr);
      llvm::SmallPtrSet<Operation*,1> ptrSet;
      ptrSet.insert(when1Op.getOperation());
      fbyOp.getResult().replaceAllUsesExcept(when1Op.getResult(),
					     ptrSet);

      OperationState when2State(loc, WhenOp::getOperationName());
      WhenOp::build(builder, when2State, condNot, retFby);
      Operation* when2OpPtr = builder.createOperation(when2State);
      WhenOp when2Op = dyn_cast<WhenOp>(when2OpPtr);

      Value rhsFby = fbyOp.getRhs();
      OperationState mergeState(loc, MergeOp::getOperationName());
      MergeOp::build(builder, mergeState, cond, rhsFby, when2Op.getResult());
      Operation* mergeOpPtr = builder.createOperation(mergeState);
      MergeOp mergeOp = dyn_cast<MergeOp>(mergeOpPtr);

      op->setOperand(1,mergeOp.getResult());
      
      return;
    }
  }
}
