// lus/LusCond.h - Custom passes defnitions -*- C++ -*- //

#ifndef MLIRLUS_PASSES_H
#define MLIRLUS_PASSES_H

#include <memory>

namespace mlir {
  class Pass;
  std::unique_ptr<mlir::Pass> createUpOpsPass();
  namespace lus {
    std::unique_ptr<mlir::Pass> createPersistFbyPass();
    std::unique_ptr<mlir::Pass> createRemoveFbyPass();
    std::unique_ptr<mlir::Pass> createRemovePrePass();
    std::unique_ptr<mlir::Pass> createScheduleDominancePass();
    std::unique_ptr<mlir::Pass> createGenPredicatesPass();
    std::unique_ptr<mlir::Pass> createGenCondactsPass();
    std::unique_ptr<mlir::Pass> createGenOutputsPass();
    std::unique_ptr<mlir::Pass> createInlineNodesPass();
    std::unique_ptr<mlir::Pass> createNodeToFunPass();
    std::unique_ptr<mlir::Pass> createLusToSyncPass();
    std::unique_ptr<mlir::Pass> createLusToPssaPass();
  }
  namespace pssa {
    std::unique_ptr<mlir::Pass> createFusionCondactsPass();
    std::unique_ptr<mlir::Pass> createPssaToStandardPass();
    std::unique_ptr<mlir::Pass> createPssaToLLVMPass();
  }
  namespace sync {
    std::unique_ptr<mlir::Pass> createSyncToStandardPass();
    std::unique_ptr<mlir::Pass> createSyncToLLVMPass();
  }
}

#endif
