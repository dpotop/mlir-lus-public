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

namespace mlir {
  namespace lus {

    struct NodeToNodePass : public PassWrapper< NodeToNodePass,
						OperationPass<ModuleOp>> {
      void runOnOperation() override;
    };
      
    void NodeToNodePass::runOnOperation() {
      auto mod = getOperation();

      sync::NodeToNode nodeToNode;
      nodeToNode(mod);
      for (Operation &op : *(mod.getBodyRegion().begin())) {
	if (isa<sync::NodeOp>(&op)) {
	  sync::NodeOp syncNodeOp = dyn_cast<sync::NodeOp>(&op);
	  pssa::LowerPssa lowerPssa;
	  lowerPssa(syncNodeOp);
	}
      }
    }

    std::unique_ptr<Pass> createNodeToNodePass() {
      return std::make_unique<NodeToNodePass>();
    }
    
  }
}
