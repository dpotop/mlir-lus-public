//===- LoopUtils.cpp ---- Misc utilities for loop transformation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous loop transformation routines.
//
//===----------------------------------------------------------------------===//

#include "PermuteLoops.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "PermuteLoops"

using namespace mlir;
using llvm::SetVector;
using llvm::SmallMapVector;

namespace {
// This structure is to pass and return sets of loop parameters without
// confusing the order.
struct LoopParams {
  Value lowerBound;
  Value upperBound;
  Value step;
};
} // namespace


// input[i] should move from position i -> permMap[i]. Returns the position in
// `input` that becomes the new outermost loop.
unsigned mlir::prime::permuteLoops(MutableArrayRef<AffineForOp> input,
				   ArrayRef<unsigned> permMap) {
  // std::error_code err;
  // llvm::raw_fd_ostream stream("/dev/stdout", err);
  // stream << "\ninput size = " << input.size() << " & map size = " << permMap.size() << "\n";
  if(input.size() != permMap.size())
    return 0;
  // Check whether the permutation spec is valid. This is a small vector - we'll
  // just sort and check if it's iota.
  SmallVector<unsigned, 4> checkPermMap(permMap.begin(), permMap.end());
  llvm::sort(checkPermMap);
  if (llvm::any_of(llvm::enumerate(checkPermMap),
                   [](const auto &en) { return en.value() != en.index(); }))
    assert(false && "invalid permutation map");

  // Nothing to do.
  if (input.size() < 2)
    return 0;

  assert(isPerfectlyNested(input) && "input not perfectly nested");

  // Compute the inverse mapping, invPermMap: since input[i] goes to position
  // permMap[i], position i of the permuted nest is at input[invPermMap[i]].
  SmallVector<std::pair<unsigned, unsigned>, 4> invPermMap;
  for (unsigned i = 0, e = input.size(); i < e; ++i)
    invPermMap.push_back({permMap[i], i});
  llvm::sort(invPermMap);

  // Move the innermost loop body to the loop that would be the innermost in the
  // permuted nest (only if the innermost loop is going to change).
  if (permMap.back() != input.size() - 1) {
    auto *destBody = input[invPermMap.back().second].getBody();
    auto *srcBody = input.back().getBody();
    destBody->getOperations().splice(destBody->begin(),
                                     srcBody->getOperations(), srcBody->begin(),
                                     std::prev(srcBody->end()));
  }

  // We'll move each loop in `input` in the reverse order so that its body is
  // empty when we are moving it; this incurs zero copies and no erasing.
  for (int i = input.size() - 1; i >= 0; --i) {
    // If this has to become the outermost loop after permutation, add it to the
    // parent block of the original root.
    if (permMap[i] == 0) {
      // If the root remains the same, nothing to do.
      if (i == 0)
        continue;
      // Make input[i] the new outermost loop moving it into parentBlock.
      auto *parentBlock = input[0]->getBlock();
      parentBlock->getOperations().splice(Block::iterator(input[0]),
                                          input[i]->getBlock()->getOperations(),
                                          Block::iterator(input[i]));
      continue;
    }

    // If the parent in the permuted order is the same as in the original,
    // nothing to do.
    unsigned parentPosInInput = invPermMap[permMap[i] - 1].second;
    if (i > 0 && static_cast<unsigned>(i - 1) == parentPosInInput)
      continue;

    // Move input[i] to its surrounding loop in the transformed nest.
    auto *destBody = input[parentPosInInput].getBody();
    destBody->getOperations().splice(destBody->begin(),
                                     input[i]->getBlock()->getOperations(),
                                     Block::iterator(input[i]));
  }

  return invPermMap[0].second;
}

bool LLVM_ATTRIBUTE_UNUSED
mlir::prime::isPerfectlyNested(ArrayRef<AffineForOp> loops) {
  assert(!loops.empty() && "no loops provided");

  // We already know that the block can't be empty.
  auto hasTwoElements = [](Block *block) {
    auto secondOpIt = std::next(block->begin());
    return secondOpIt != block->end() && &*secondOpIt == &block->back();
  };

  auto enclosingLoop = loops.front();
  for (auto loop : loops.drop_front()) {
    auto parentForOp = dyn_cast<AffineForOp>(loop->getParentOp());
    // parentForOp's body should be just this loop and the terminator.
    if (parentForOp != enclosingLoop || !hasTwoElements(parentForOp.getBody()))
      return false;
    enclosingLoop = loop;
  }
  return true;
}

template <typename T>
static void getPerfectlyNestedLoopsImpl(
    SmallVectorImpl<T> &forOps, T rootForOp,
    unsigned maxLoops = std::numeric_limits<unsigned>::max()) {
  for (unsigned i = 0; i < maxLoops; ++i) {
    forOps.push_back(rootForOp);
    Block &body = rootForOp.region().front();
    if (body.begin() != std::prev(body.end(), 2))
      return;

    rootForOp = dyn_cast<T>(&body.front());
    if (!rootForOp)
      return;
  }
}

/// Get perfectly nested sequence of loops starting at root of loop nest
/// (the first op being another AffineFor, and the second op - a terminator).
/// A loop is perfectly nested iff: the first op in the loop's body is another
/// AffineForOp, and the second op is a terminator).
void mlir::prime::getPerfectlyNestedLoops(SmallVectorImpl<AffineForOp> &nestedLoops,
                                   AffineForOp root) {
  getPerfectlyNestedLoopsImpl(nestedLoops, root);
}

void mlir::prime::getPerfectlyNestedLoops(SmallVectorImpl<scf::ForOp> &nestedLoops,
                                   scf::ForOp root) {
  getPerfectlyNestedLoopsImpl(nestedLoops, root);
}
