#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "../../Dialects/Sync/UndefOp.h"

namespace mlir {
  namespace sync {

    class UndefLowering : public ConversionPattern {
    public:
      
      explicit UndefLowering(MLIRContext *context)
	: ConversionPattern(UndefOp::getOperationName(), 1, context) {}

      LogicalResult matchAndRewrite(Operation *op,
				    ArrayRef<Value> operands,
				    ConversionPatternRewriter &rewriter) const override {
	UndefOp undefOp = dyn_cast<UndefOp>(op);
	auto loc = undefOp.getLoc();

	Type nt = undefOp.getResult().getType();
	Operation *llvmUndefOp = rewriter.create<LLVM::UndefOp>(loc, nt);
	rewriter.replaceOp(undefOp, llvmUndefOp->getResults());
	return success();
      }
    };
    
    struct SyncToLLVMPass
      : public PassWrapper<SyncToLLVMPass, OperationPass<ModuleOp>> {
      void getDependentDialects(DialectRegistry &registry) const override {
	registry.insert<LLVM::LLVMDialect>();
      }
      void runOnOperation() final;
    };

    void SyncToLLVMPass::runOnOperation() {

      auto mod = getOperation();
      
      LLVMConversionTarget target(getContext());
      target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
      target.addIllegalOp<UndefOp>();

      LLVMTypeConverter typeConverter(&getContext());

      OwningRewritePatternList patterns(&getContext());
      populateStdToLLVMConversionPatterns(typeConverter, patterns);
      populateVectorToLLVMConversionPatterns(typeConverter, patterns);

      patterns.insert<UndefLowering>(&getContext());

      if (failed(applyPartialConversion(mod, target, std::move(patterns))))
	signalPassFailure();
    }

    std::unique_ptr<Pass> createSyncToLLVMPass() {
      return std::make_unique<SyncToLLVMPass>();
    }
  }
}
