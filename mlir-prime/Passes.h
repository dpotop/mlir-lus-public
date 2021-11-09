// lus/LusCond.h - Custom passes defnitions -*- C++ -*- //

#ifndef PRIME_PASSES_H
#define PRIME_PASSES_H

#include <memory>

namespace mlir {
  class Pass;
  void registerLoopPermutationPrimePass();
  void registerBufferizeReshapePass();
  void registerRemoveCopyPrimePass();
  void registerPrimeLinalgToAffinePass();
  // std::unique_ptr<mlir::Pass> createUpOpsPass();
}

#endif
