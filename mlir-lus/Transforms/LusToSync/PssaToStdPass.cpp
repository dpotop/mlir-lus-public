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
#include "../SyncToStd/RemoveSomeUndefs.h"
#include "../SyncToStd/RemovePsis.h"

namespace mlir {
  namespace pssa {

    struct PssaToStandardPass : public PassWrapper< PssaToStandardPass,
						    OperationPass<ModuleOp>> {
      void runOnOperation() override;
    };
      
    void PssaToStandardPass::runOnOperation() {
      
      auto mod = getOperation();

      RemovePsis removePsis;
      removePsis(mod);
      
    }

    std::unique_ptr<Pass> createPssaToStandardPass() {
      return std::make_unique<PssaToStandardPass>();
    }
    
  }
}
