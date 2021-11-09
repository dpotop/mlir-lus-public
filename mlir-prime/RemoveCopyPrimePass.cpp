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
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "PermuteLoops.h"
#include "Passes.h"
#include <list>
#include <utility>

#define PASS_NAME "remove-copy-prime"

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory(PASS_NAME " options");

namespace {

struct RemoveCopyPrimePass
    : public PassWrapper<RemoveCopyPrimePass, FunctionPass> {
  RemoveCopyPrimePass() = default;
  RemoveCopyPrimePass(const RemoveCopyPrimePass &pass){};

  void runOnFunction() override;

};

}

void RemoveCopyPrimePass::runOnFunction() {

  std::error_code err;
  llvm::raw_fd_ostream stream("/dev/stdout", err);
  
  std::list<linalg::CopyOp> copyOps;
  for (auto &op : getFunction().front()) {
    if (auto linalgCopyOp = dyn_cast<linalg::CopyOp>(op)) {
	copyOps.push_back(linalgCopyOp);
      }
      
      // reshapeOps.push_back(tensorReshapeOp);
  }

  for (auto linalgCopyOp : copyOps) {
    Value input = linalgCopyOp.input();
    Value output = linalgCopyOp.output();
    MemRefType outputType = output.getType().cast<MemRefType>();
    Region *parentRegion = output.getParentRegion();
    bool isArg = false;
    for (Value v : parentRegion->getArguments()) {
      if (output == v) {
	isArg = true;
	break;
      }
    }
    bool hasAffineMaps = (outputType.getAffineMaps().size() > 0);
    if (!hasAffineMaps && !isArg) {
      output.replaceAllUsesWith(input);
      linalgCopyOp.erase();
    }
  }
  // for (auto tensorReshapeOp: reshapeOps) {

  //   Location loc = tensorReshapeOp.getOperation()->getLoc();
  //   OpBuilder builder(tensorReshapeOp);

  //   Value src = tensorReshapeOp.src();
  //   TensorType srcTensorTy = src.getType().cast<TensorType>();
  //   MemRefType srcMemrefTy = MemRefType::get(srcTensorTy.getShape(),
  // 					     srcTensorTy.getElementType());

  //   Value dest = tensorReshapeOp.result();
  //   TensorType destTensorTy = dest.getType().cast<TensorType>();
  //   MemRefType destMemrefTy = MemRefType::get(destTensorTy.getShape(),
  // 					      destTensorTy.getElementType());

  //   memref::BufferCastOp bufferCastOp = builder.create<memref::BufferCastOp>(loc, srcMemrefTy, src);
  //   Value buffer = bufferCastOp.memref();
  //   linalg::ReshapeOp reshapeOp = builder.create<linalg::ReshapeOp>(loc, destMemrefTy, buffer,
  // 								    tensorReshapeOp.reassociation());
  //   Value reshaped = reshapeOp.result();
  //   memref::TensorLoadOp tensorLoadOp = builder.create<memref::TensorLoadOp>(loc, reshaped);
  //   Value res = tensorLoadOp.result();
  //   dest.replaceAllUsesWith(res);

  // }

  // for (auto tensorReshapeOp: reshapeOps) {
  //   tensorReshapeOp.erase();
  // }
}

namespace mlir {
void registerRemoveCopyPrimePass() {
  PassRegistration<RemoveCopyPrimePass>(
      PASS_NAME, "Remove useless linalg.copy");
}
} // namespace mlir
