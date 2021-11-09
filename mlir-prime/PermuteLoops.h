
#include "mlir/IR/Block.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"



namespace mlir {

  namespace scf {
    class ForOp;
    class ParallelOp;
  }
  
  class AffineForOp;
  class AffineMap;
  class FuncOp;
  class LoopLikeOpInterface;
  struct MemRefRegion;
  class OpBuilder;
  class Value;
  class ValueRange;
  
  namespace prime {
    unsigned permuteLoops(MutableArrayRef<AffineForOp> input,
			  ArrayRef<unsigned> permMap);
    bool LLVM_ATTRIBUTE_UNUSED isPerfectlyNested(ArrayRef<AffineForOp> loops);
    void getPerfectlyNestedLoops(SmallVectorImpl<AffineForOp> &nestedLoops,
                             AffineForOp root);
    void getPerfectlyNestedLoops(SmallVectorImpl<scf::ForOp> &nestedLoops,
				 scf::ForOp root);
  }
}
