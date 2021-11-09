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

#define PASS_NAME "bufferize-linalg-reshape"

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory(PASS_NAME " options");

namespace {

struct BufferizeReshapePass
    : public PassWrapper<BufferizeReshapePass, FunctionPass> {
  BufferizeReshapePass() = default;
  BufferizeReshapePass(const BufferizeReshapePass &pass){};

  void runOnFunction() override;

};

}

void grabReshapeOps(std::list<linalg::TensorReshapeOp> &reshapeOps, Operation & op) {
  if (auto tensorReshapeOp = dyn_cast<linalg::TensorReshapeOp>(op)) {
    reshapeOps.push_back(tensorReshapeOp);
  }
  else {
    for (Region &r: op.getRegions()) {
      for (Block &b: r.getBlocks()) {
	for (Operation &deeperOp: b.getOperations()) {
	  grabReshapeOps(reshapeOps, deeperOp);
	}
      }
    }
  }
}

void BufferizeReshapePass::runOnFunction() {

  std::list<linalg::TensorReshapeOp> reshapeOps;
  for (auto &op : getFunction().front()) {
    grabReshapeOps(reshapeOps, op);
    // if (auto tensorReshapeOp = dyn_cast<linalg::TensorReshapeOp>(op)) {
    //   reshapeOps.push_back(tensorReshapeOp);
    // }
  }
  for (auto tensorReshapeOp: reshapeOps) {

    Location loc = tensorReshapeOp.getOperation()->getLoc();
    OpBuilder builder(tensorReshapeOp);

    Value src = tensorReshapeOp.src();
    TensorType srcTensorTy = src.getType().cast<TensorType>();
    MemRefType srcMemrefTy = MemRefType::get(srcTensorTy.getShape(),
					     srcTensorTy.getElementType());

    Value dest = tensorReshapeOp.result();
    TensorType destTensorTy = dest.getType().cast<TensorType>();
    MemRefType destMemrefTy = MemRefType::get(destTensorTy.getShape(),
					      destTensorTy.getElementType());

    memref::BufferCastOp bufferCastOp = builder.create<memref::BufferCastOp>(loc, srcMemrefTy, src);
    Value buffer = bufferCastOp.memref();
    linalg::ReshapeOp reshapeOp = builder.create<linalg::ReshapeOp>(loc, destMemrefTy, buffer,
								    tensorReshapeOp.reassociation());
    Value reshaped = reshapeOp.result();
    memref::TensorLoadOp tensorLoadOp = builder.create<memref::TensorLoadOp>(loc, reshaped);
    Value res = tensorLoadOp.result();
    dest.replaceAllUsesWith(res);

  }

  for (auto tensorReshapeOp: reshapeOps) {
    tensorReshapeOp.erase();
  }
}

namespace mlir {
void registerBufferizeReshapePass() {
  PassRegistration<BufferizeReshapePass>(
      PASS_NAME, "Bufferize Linalg reshape");
}
} // namespace mlir
