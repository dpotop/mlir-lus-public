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
#include "../LusToSync/GenOutputs.h"

namespace mlir {
  namespace lus {

    struct GenOutputsPass : public PassWrapper< GenOutputsPass,
						 OperationPass<NodeOp>> {
      void runOnOperation() override;
    };
      
    void GenOutputsPass::runOnOperation() {
      auto lusNodeOp = getOperation();

      if (lusNodeOp.hasBody()) {
	GenOutputs genOutputs;
	genOutputs(lusNodeOp);
      }
    }

    std::unique_ptr<Pass> createGenOutputsPass() {
      return std::make_unique<GenOutputsPass>();
    }
    
  }
}
