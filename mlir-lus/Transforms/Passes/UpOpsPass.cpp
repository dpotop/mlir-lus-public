#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
// #include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include <list>

namespace mlir {

  struct UpOpsPass : public PassWrapper< UpOpsPass, OperationPass<ModuleOp>> {
    void runOnOperation() override;
  };

  static void upOps(Operation *curr, std::list<Operation*> & shouldBeMoved) {
    for (Region &r: curr->getRegions()) {
      for (Block &b: r) {
	for (Operation &op: b) {
	  upOps(&op, shouldBeMoved);
	  if (isa<ReturnOp>(&op)) {
	  }
	  else if (isa<CallOp>(&op)) {
	    CallOp callOp = dyn_cast<CallOp>(&op);
	    if (llvm::size(callOp.getArgOperands()) == 0) {
	      shouldBeMoved.push_back(&op);
	    }
	  }
	  else if (isa<ConstantOp>(&op) || op.getNumOperands() == 0) {
	    shouldBeMoved.push_back(&op);
	  }
	}
      }
    }
  }
  
  void UpOpsPass::runOnOperation() {
    Operation *moduleOpPtr = getOperation();
    ModuleOp moduleOp = dyn_cast<ModuleOp>(moduleOpPtr);
    for (Operation &op: *(moduleOp.getBodyRegion().begin())) {
      if (isa<FuncOp>(&op)) {
	FuncOp funcOp = dyn_cast<FuncOp>(&op);
	std::list<Operation *> shouldBeMoved;
	if (!funcOp.isExternal()) {
	  upOps(funcOp, shouldBeMoved);
	}
	for (Operation *move: shouldBeMoved) {
	  move->moveBefore(&funcOp.getRegion().front().front());
	}
      }
    }
  }

  std::unique_ptr<Pass> createUpOpsPass() {
    return std::make_unique<UpOpsPass>();
  }
    
}
