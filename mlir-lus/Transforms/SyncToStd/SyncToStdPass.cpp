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
#include "../../Dialects/Sync/SyncOp.h"
#include "../../Dialects/Sync/Node.h"
#include "../../Dialects/Sync/InputOp.h"
#include "../../Dialects/Sync/TickOp.h"
#include "NodeToFun.h"
#include "LowerTick.h"
#include "../SyncToStd/RemoveSomeUndefs.h"
#include "RemovePsis.h"

namespace mlir {
  namespace sync {

    struct SyncToStandardPass : public PassWrapper< SyncToStandardPass,
						    OperationPass<ModuleOp>> {
      void runOnOperation() override;
    };
      
    void SyncToStandardPass::runOnOperation() {

      auto mod = getOperation();

      NodeToFun nodeToFun;
      nodeToFun(mod);
      LowerTick lowerTick;
      lowerTick(mod);
      pssa::RemoveSomeUndefs removeSomeUndefs;
      removeSomeUndefs(mod);
      pssa::RemovePsis removePsis;
      removePsis(mod);
    }

    std::unique_ptr<Pass> createSyncToStandardPass() {
      return std::make_unique<SyncToStandardPass>();
    }
    
  }
}
