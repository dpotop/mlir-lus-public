#include <list>

#include "../../Tools/CommandLine.h"
#include "../../Dialects/Lus/lus.h"
#include "../../Dialects/Pssa/pssa.h"
#include "../LusToLus/InlineNodes.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
  namespace lus {

    class InlineNodesPass : public PassWrapper < InlineNodesPass,
					       OperationPass<ModuleOp> > {
      void runOnOperation() override;
    };

    void InlineNodesPass::runOnOperation() {
      ConversionTarget target(getContext());
      target.addLegalDialect<pssa::Pssa,
			     StandardOpsDialect
			     >();
      target.addIllegalDialect<Lus>();

      
      ModuleOp moduleOp(getOperation());
      InlineNodes inlineNodes;
      inlineNodes(moduleOp);
    }

    std::unique_ptr<Pass> createInlineNodesPass() {
      return std::make_unique<InlineNodesPass>();
    }
  }
}
	
