#include "../../Dialects/Lus/lus.h"
#include "../../Dialects/Pssa/pssa.h"
#include "../../Dialects/Lus/WhenOp.h"
#include "../../Dialects/Lus/MergeOp.h"
#include "../../Dialects/Lus/PreOp.h"
#include "../LusToLus/RemoveFby.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace std;

namespace mlir {
  namespace lus {

    class RemoveFbyPass : public PassWrapper<
      RemoveFbyPass,
      OperationPass<NodeOp>> {
      void runOnOperation() override;
    };

    void RemoveFbyPass::runOnOperation() {
      ConversionTarget target(getContext());
      target.addLegalDialect<pssa::Pssa,
			     StandardOpsDialect
			     >();
      target.addIllegalDialect<Lus>();
      target.addLegalOp <
	WhenOp,
	MergeOp,
	NodeOp,
	PreOp,
	lus::YieldOp
	>();

      Operation* op = getOperation();
      NodeOp nodeOp = dyn_cast<NodeOp>(op);
      if (!nodeOp.hasBody())
	return;
      RemoveFby removeFby;
      removeFby(nodeOp);
    }

    std::unique_ptr<Pass> createRemoveFbyPass() {
      return std::make_unique<RemoveFbyPass>();
    }
  }
}
