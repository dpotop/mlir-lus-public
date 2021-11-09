#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/IR/Dialect.h"
#include "Passes.h"

using namespace mlir ;
// using namespace mlir::vector;


int main(int argc, char **argv) {
  registerAllPasses();
  registerLoopPermutationPrimePass();
  registerBufferizeReshapePass();
  registerRemoveCopyPrimePass();
  registerPrimeLinalgToAffinePass();
  DialectRegistry registry;
  registerAllDialects(registry);
  return failed(MlirOptMain(argc, argv, "MLIR modular optimizer driver\n",
                            registry,
                            /*preloadDialectsInContext=*/false));
}
