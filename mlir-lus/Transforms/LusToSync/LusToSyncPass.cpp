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
#include "../../Dialects/Lus/Node.h"
#include "../../Dialects/Pssa/pssa.h"
#include "GenPredicates.h"
#include "GenCondacts.h"
#include "FusionCondacts.h"
#include "NodeToNode.h"
#include "LowerPssa.h"
#include "LowerInst.h"

namespace mlir {
  namespace lus {

    struct LusToSyncPass : public PassWrapper< LusToSyncPass,
						OperationPass<ModuleOp>> {
      void runOnOperation() override;
    };
      
    void LusToSyncPass::runOnOperation() {

      auto mod = getOperation();

      sync::NodeToNode nodeToNode;
      nodeToNode(mod);
      sync::LowerInst lowerInst;
      lowerInst(mod);
      list<Operation*> topLevel;
      for (Operation &op : *(mod.getBodyRegion().begin())) {
      	topLevel.push_back(&op);
      }
      for (Operation *op : topLevel) {
	if (isa<sync::NodeOp>(op)) {
	  sync::NodeOp syncNodeOp = dyn_cast<sync::NodeOp>(op);
	  pssa::LowerPssa lowerPssa;
	  lowerPssa(syncNodeOp);
	}
      }
    }

    std::unique_ptr<Pass> createLusToSyncPass() {
      return std::make_unique<LusToSyncPass>();
    }
    
  }
}
