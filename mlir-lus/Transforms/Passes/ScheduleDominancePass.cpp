#include "../../Dialects/Lus/lus.h"
#include "../../Dialects/Pssa/pssa.h"
#include "../../Dialects/Lus/WhenOp.h"
#include "../../Dialects/Lus/MergeOp.h"
#include "../LusToLus/EnsureDominance.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace std;

namespace mlir {
  namespace lus {

    class ScheduleDominancePass : public PassWrapper<
      ScheduleDominancePass,
      OperationPass<NodeOp>> {
      void runOnOperation() override;
    };

    void ScheduleDominancePass::runOnOperation() {
      ConversionTarget target(getContext());
      target.addLegalDialect<pssa::Pssa,
			     StandardOpsDialect
			     >();
      target.addIllegalDialect<Lus>();
      target.addLegalOp <
	WhenOp,
	MergeOp,
	NodeOp,
	lus::YieldOp
	>();

      Operation* op = getOperation();
      NodeOp nodeOp = dyn_cast<NodeOp>(op);
      if (!nodeOp.hasBody())
	return;
      EnsureDominance ensureDominance;
      ensureDominance(nodeOp);
    }

    std::unique_ptr<Pass> createScheduleDominancePass() {
      return std::make_unique<ScheduleDominancePass>();
    }
  }
}
