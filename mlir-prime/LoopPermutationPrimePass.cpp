//===- TestLoopPermutation.cpp - Test affine loop permutation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test the affine for op permutation utility.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "PermuteLoops.h"
#include "Passes.h"
#include <list>
#include <utility>

#define PASS_NAME "loop-permutation-prime"

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory(PASS_NAME " options");

namespace {

/// This pass applies the permutation on the first maximal perfect nest.
struct LoopPermutationPrimePass
    : public PassWrapper<LoopPermutationPrimePass, FunctionPass> {
  LoopPermutationPrimePass() = default;
  LoopPermutationPrimePass(const LoopPermutationPrimePass &pass){};

  void runOnFunction() override;

private:
  /// Permutation specifying loop i is mapped to permList[i] in
  /// transformed nest (with i going from outermost to innermost).
  ListOption<unsigned> permList{*this, "permutation-map",
                                llvm::cl::desc("Specify the loop permutation"),
                                llvm::cl::OneOrMore, llvm::cl::CommaSeparated};
};

} // end anonymous namespace

void LoopPermutationPrimePass::runOnFunction() {
  // Get the first maximal perfect nest.
  std::list<std::pair<SmallVector<AffineForOp, 6>, SmallVector<unsigned, 4>>> allPerms;
  for (auto &op : getFunction().front()) {
    if (auto forOp = dyn_cast<AffineForOp>(op)) {
      SmallVector<AffineForOp, 6> nest;
      mlir::prime::getPerfectlyNestedLoops(nest, forOp);
      SmallVector<unsigned, 4> permMap(permList.begin(), permList.end());
      std::pair<SmallVector<AffineForOp, 6>, SmallVector<unsigned, 4>> p(nest, permMap);
      allPerms.push_back(p);
      // mlir::prime::permuteLoops(nest, permMap);
    }
  }
  for (std::pair<SmallVector<AffineForOp, 6>, SmallVector<unsigned, 4>> p : allPerms) {
    mlir::prime::permuteLoops(p.first, p.second);
  }
}

namespace mlir {
void registerLoopPermutationPrimePass() {
  PassRegistration<LoopPermutationPrimePass>(
      PASS_NAME, "Tests affine loop permutation utility");
}
} // namespace mlir
