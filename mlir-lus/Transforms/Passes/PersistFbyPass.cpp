#include "../../Dialects/Lus/lus.h"
#include "../../Dialects/Pssa/pssa.h"
#include "../../Dialects/Lus/WhenOp.h"
#include "../../Dialects/Lus/MergeOp.h"
#include "../../Dialects/Lus/PreOp.h"
#include "../LusToLus/PersistFby.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace std;

namespace mlir {
  namespace lus {

    class PersistFbyPass : public PassWrapper<
      PersistFbyPass,
      OperationPass<NodeOp>> {
      void runOnOperation() override;
    };

    void PersistFbyPass::runOnOperation() {
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
      PersistFby persistFby;
      persistFby(nodeOp);
    }

    std::unique_ptr<Pass> createPersistFbyPass() {
      return std::make_unique<PersistFbyPass>();
    }
  }
}
