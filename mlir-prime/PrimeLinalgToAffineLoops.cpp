#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "Passes.h"
#include <list>
#include <utility>

#include <iostream>

#define PASS_NAME "prime-linalg-to-affine"

using namespace mlir;

namespace {
  struct PrimeLinalgToAffinePass
    : public PassWrapper<PrimeLinalgToAffinePass, FunctionPass> {
    PrimeLinalgToAffinePass() = default;
    PrimeLinalgToAffinePass(const PrimeLinalgToAffinePass &pass){};
    void runOnFunction() override;
  };
}

const int64_t TS_I = 7;
const int64_t TS_J = 8;
const int64_t TS_K = 64;

Value packAt(OpBuilder &builder, Location loc,
	     const int64_t SI1, const int SI2,
	     Value zeroConstant, Value input) {

  MemRefType inputType = input.getType().cast<MemRefType>();
  Type eltType = inputType.getElementType();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  const int64_t I1 = inputShape[1];
  const int64_t I2 = inputShape[2];
  const int64_t K = inputShape[3];

  int64_t atShape[4];
  atShape[0] = 1;
  atShape[1] = I1/SI1;
  atShape[2] = I2/SI2;
  atShape[3] = K;
  MemRefType atType = MemRefType::get(atShape, eltType);

  memref::AllocOp allocAtOp = builder.create<memref::AllocOp>(loc, atType);
  Value at = allocAtOp.memref();

  AffineForOp i1LoopOp = builder.create<AffineForOp>(loc, 0, I1/SI1);
  Value i1Var = i1LoopOp.getInductionVar();
  builder.setInsertionPoint(&i1LoopOp.region().front(),
			    i1LoopOp.region().front().begin());
  AffineForOp i2LoopOp = builder.create<AffineForOp>(loc, 0, I2/SI2);
  Value i2Var = i2LoopOp.getInductionVar();
  builder.setInsertionPoint(&i2LoopOp.region().front(),
			    i2LoopOp.region().front().begin());
  AffineForOp kLoopOp = builder.create<AffineForOp>(loc, 0, K);
  Value kVar = kLoopOp.getInductionVar();
  builder.setInsertionPoint(&kLoopOp.region().front(),
			    kLoopOp.region().front().begin());

  auto identity0Expr = builder.getAffineDimExpr(0);
  auto dim1Expr = builder.getAffineDimExpr(1) * SI1;
  auto dim2Expr = builder.getAffineDimExpr(2) * SI2;
  auto identity3Expr = builder.getAffineDimExpr(3);
  AffineMap inputAM = AffineMap::get(4, 0,
				    {identity0Expr, dim1Expr, dim2Expr, identity3Expr},
				    builder.getContext());
  OperationState inputLoadState(loc, AffineLoadOp::getOperationName());
  AffineLoadOp::build(builder, inputLoadState, input,
		      inputAM, {zeroConstant, i1Var, i2Var, kVar});
  Operation *inputLoadOpPtr = builder.createOperation(inputLoadState);
  AffineLoadOp inputLoadOp = dyn_cast<AffineLoadOp>(inputLoadOpPtr);
  Value elt = inputLoadOp.result();

  auto identity1Expr = builder.getAffineDimExpr(1);
  auto identity2Expr = builder.getAffineDimExpr(2);
  AffineMap outputAM = AffineMap::get(4, 0,
				      {identity0Expr, identity1Expr, identity2Expr, identity3Expr },
				      builder.getContext());
  OperationState outputStoreState(loc,
				  AffineStoreOp::getOperationName());
  AffineStoreOp::build(builder, outputStoreState,
		       elt, at, outputAM,
		       {zeroConstant, i1Var, i2Var, kVar});
  builder.createOperation(outputStoreState);
  
  
  return at;
}

void baseCase(OpBuilder &builder, Location loc,
	      Value zeroConstant,
	      Value input, Value kernel, Value output) {

  AffineMap identityMap = builder.getDimIdentityMap();

  ArrayRef<int64_t> kernelShape = kernel.getType().cast<MemRefType>().getShape();
  Attribute JAttr = IntegerAttr::get(builder.getIndexType(), kernelShape[3]);
  ConstantOp JConstantOp = builder.create<ConstantOp>(loc, JAttr);
  Value JConstant = JConstantOp.getResult();
  ArrayRef<int64_t> inputShape = input.getType().cast<MemRefType>().getShape();
  Attribute KAttr = IntegerAttr::get(builder.getIndexType(), inputShape[3]);
  ConstantOp KConstantOp = builder.create<ConstantOp>(loc, KAttr);
  Value KConstant = KConstantOp.getResult();
  Attribute I1Attr = IntegerAttr::get(builder.getIndexType(), inputShape[1]);
  ConstantOp I1ConstantOp = builder.create<ConstantOp>(loc, I1Attr);
  Value I1Constant = I1ConstantOp.getResult();
  Type outputElementType = output.getType().cast<MemRefType>().getElementType();
  
  int64_t btShape[2];
  btShape[0] = TS_K;
  btShape[1] = TS_J;
  MemRefType btType = MemRefType::get(btShape, outputElementType);
  memref::AllocaOp btOp = builder.create<memref::AllocaOp> (loc, btType);
  Value bt = btOp.memref();

  // k tile

  auto divByTSKExpr = builder.getAffineDimExpr(0).floorDiv(TS_K);
  AffineMap divByTSKAM =  AffineMap::get(/*dimCount=*/1,
  					     /*symbolCount=*/0,
  					    divByTSKExpr);

  OperationState kTileState(loc, AffineForOp::getOperationName());
  AffineForOp::build(builder, kTileState,
  			 {zeroConstant}, identityMap,
  			 {KConstant}, divByTSKAM);
  Operation* kTileLoopPtr = builder.createOperation(kTileState);
  AffineForOp kTileLoop = dyn_cast<AffineForOp>(kTileLoopPtr);
  Value kTile = kTileLoop.getInductionVar();
  builder.setInsertionPoint(&kTileLoop.region().front(),
			    kTileLoop.region().front().begin());
  
  // j tile

  auto divByTSJExpr = builder.getAffineDimExpr(0).floorDiv(TS_J);
  AffineMap divByTSJAM =  AffineMap::get(/*dimCount=*/1,
					 /*symbolCount=*/0,
					 divByTSJExpr);

  OperationState jTileState(loc, AffineForOp::getOperationName());
  AffineForOp::build(builder, jTileState,
		     {zeroConstant}, identityMap,
		     {JConstant}, divByTSJAM);
  Operation *jTileLoopPtr = builder.createOperation(jTileState);
  AffineForOp jTileLoop = dyn_cast<AffineForOp>(jTileLoopPtr);
  Value jTile = jTileLoop.getInductionVar();
  builder.setInsertionPoint(&jTileLoop.region().front(),
			    jTileLoop.region().front().begin());
  
  // pack
  
  AffineForOp kLoopPack = builder.create<AffineForOp>(loc, 0, TS_K);
  builder.setInsertionPoint(&kLoopPack.region().front(),
			    kLoopPack.region().front().begin());
  AffineForOp jLoopPack = builder.create<AffineForOp>(loc, 0, TS_J);
  builder.setInsertionPoint(&jLoopPack.region().front(),
			    jLoopPack.region().front().begin());
      
	
  Value kPack = kLoopPack.getInductionVar();
  Value jPack = jLoopPack.getInductionVar();

  auto identity0Expr = builder.getAffineDimExpr(0);
  auto dimKExpr = builder.getAffineDimExpr(1)
    + (builder.getAffineDimExpr(2) * TS_K);
  auto dimJExpr = builder.getAffineDimExpr(3)
    + (builder.getAffineDimExpr(4) * TS_J);
  AffineMap packAM = AffineMap::get(5, 0,
				    {identity0Expr, identity0Expr, dimKExpr, dimJExpr},
				    builder.getContext());

  OperationState packLoadState(loc, AffineLoadOp::getOperationName());
  AffineLoadOp::build(builder, packLoadState, kernel,
		      packAM, {zeroConstant, kPack, kTile, jPack, jTile});
  Operation *packLoadOpPtr = builder.createOperation(packLoadState);
  AffineLoadOp packLoadOp = dyn_cast<AffineLoadOp>(packLoadOpPtr);
  Value elt = packLoadOp.result();

  OperationState storeState(loc, AffineStoreOp::getOperationName());
  AffineStoreOp::build(builder, storeState, elt, bt, {kPack,jPack});
  builder.createOperation(storeState);
  
  // inner loops

  builder.setInsertionPointAfter(kLoopPack);
	
  auto divByTSIExpr = builder.getAffineDimExpr(0).floorDiv(TS_I);
  AffineMap divByTSIAM =  AffineMap::get(/*dimCount=*/1,
					 /*symbolCount=*/0,
					 divByTSIExpr);

  OperationState iTileState(loc, AffineForOp::getOperationName());
  AffineForOp::build(builder, iTileState,
		     {zeroConstant}, builder.getDimIdentityMap(),
		     {I1Constant}, divByTSIAM);
  Operation* iTileLoopPtr = builder.createOperation(iTileState);
  AffineForOp iTileLoop = dyn_cast<AffineForOp>(iTileLoopPtr);
  Value iTile = iTileLoop.getInductionVar();
  builder.setInsertionPoint(&iTileLoop.region().front(),
			    iTileLoop.region().front().begin());

  AffineForOp kInLoop = builder.create<AffineForOp>(loc, 0, TS_K, TS_J);
  Value kIn = kInLoop.getInductionVar();
  builder.setInsertionPoint(&kInLoop.region().front(),
			    kInLoop.region().front().begin());
	
  AffineForOp jInLoop = builder.create<AffineForOp>(loc, 0, TS_J);
  Value jIn = jInLoop.getInductionVar();
  builder.setInsertionPoint(&jInLoop.region().front(),
			    jInLoop.region().front().begin());

  // Hoisted loads
  
  Value btLoads[TS_J];
  for (int k = 0; k < TS_J; k++) {
    auto dimKIncrLeftExpr = builder.getAffineDimExpr(0) + k;
    auto identity1Expr = builder.getAffineDimExpr(1);
    AffineMap loadBtAM = AffineMap::get(2, 0,
					{dimKIncrLeftExpr,
					 identity1Expr},
					builder.getContext());
    OperationState loadBtState(loc, AffineLoadOp::getOperationName());
    AffineLoadOp::build(builder, loadBtState, bt,
			loadBtAM, {kIn, jIn});
    Operation *loadBtOpPtr = builder.createOperation(loadBtState);
    AffineLoadOp loadBtOp = dyn_cast<AffineLoadOp>(loadBtOpPtr);
    btLoads[k] = loadBtOp.result();
  }

  // inner loops (again)
  
  AffineForOp i1InLoop = builder.create<AffineForOp>(loc, 0, TS_I);
  Value i1In = i1InLoop.getInductionVar();
  builder.setInsertionPoint(&i1InLoop.region().front(),
			    i1InLoop.region().front().begin());

  AffineForOp i2InLoop = builder.create<AffineForOp>(loc, 0, inputShape[2]);
  Value i2In = i2InLoop.getInductionVar();
  builder.setInsertionPoint(&i2InLoop.region().front(),
			    i2InLoop.region().front().begin());

  // Actual matmul
  
  auto dimInIExpr = builder.getAffineDimExpr(1)
    + (builder.getAffineDimExpr(2) * TS_I);
  auto identity3Expr = builder.getAffineDimExpr(3);
  auto dimInJExpr = builder.getAffineDimExpr(4)
    + (builder.getAffineDimExpr(5) * TS_J);

  AffineMap comingAM = AffineMap::get(6, 0,
				      {identity0Expr, dimInIExpr, identity3Expr, dimInJExpr},
				      builder.getContext());
  OperationState comingLoadState(loc, AffineLoadOp::getOperationName());
  AffineLoadOp::build(builder, comingLoadState, output,
		      comingAM, {zeroConstant, i1In, iTile, i2In, jIn, jTile});
  Operation *comingOpPtr = builder.createOperation(comingLoadState);
  AffineLoadOp comingOp = dyn_cast<AffineLoadOp>(comingOpPtr);
  Value coming = comingOp.result();

  Value acc = coming;
  for (int k = 0; k < TS_J; k++) {
	  
    auto dimKIncrRightExpr = builder.getAffineDimExpr(4) + k;
    AffineMap loadAAM =  AffineMap::get(5, 0,
					{identity0Expr, dimInIExpr, identity3Expr, dimKIncrRightExpr},
					builder.getContext());
    OperationState loadAState(loc, AffineLoadOp::getOperationName());
    AffineLoadOp::build(builder, loadAState, input,
			loadAAM, {zeroConstant, i1In, iTile, i2In, kIn});
    Operation *loadAOpPtr = builder.createOperation(loadAState);
    AffineLoadOp loadAOp = dyn_cast<AffineLoadOp>(loadAOpPtr);
    Value aElt = loadAOp.result();

    Value btElt = btLoads[k];

    MulFOp mulfOp = builder.create<MulFOp>(loc, aElt, btElt);
    AddFOp addfOp = builder.create<AddFOp>(loc, mulfOp.result(), acc);
    acc = addfOp.result();
  }

  OperationState outputStoreState(loc,
				  AffineStoreOp::getOperationName());
  AffineStoreOp::build(builder, outputStoreState,
		       acc, output, comingAM,
		       {zeroConstant, i1In, iTile, i2In, jIn, jTile});
  builder.createOperation(outputStoreState);

}

const int64_t MM_I = 3;
const int64_t MM_J = 10;
const int64_t MM_K = 50;
const int64_t CACHE_SIZE = 8*64;

void lowerMatmul(linalg::MatmulOp matmulOp) {

  OpBuilder builder(matmulOp);
  Location loc = matmulOp.getLoc();
  
  Value A = matmulOp.inputs()[0];
  Value B = matmulOp.inputs()[1];
  Value C = matmulOp.outputs()[0];

  ArrayRef<int64_t> AShape = A.getType().cast<MemRefType>().getShape();
  ArrayRef<int64_t> BShape = B.getType().cast<MemRefType>().getShape();
  ArrayRef<int64_t> CShape = C.getType().cast<MemRefType>().getShape();

  const int64_t D_I = CShape[0];
  const int64_t D_J = CShape[1];
  const int64_t D_K = BShape[0];
  const bool PACK_TRANSPOSE = D_K < CACHE_SIZE; 
  
  if (D_I != MM_I || D_K % MM_K !=0 || D_K == 1) return;

  Attribute zeroAttr = IntegerAttr::get(builder.getIndexType(), 0);
  ConstantOp zeroConstantOp = builder.create<ConstantOp>(loc, zeroAttr);
  Value zeroConstant = zeroConstantOp.getResult();

  Type eltType = A.getType().cast<MemRefType>().getElementType();

  AffineForOp k0LoopOp = builder.create<AffineForOp>(loc, 0, D_K/MM_K);
  Value k0Var = k0LoopOp.getInductionVar();
  builder.setInsertionPoint(&k0LoopOp.region().front(),
			    k0LoopOp.region().front().begin());
  
  AffineForOp j0LoopOp = builder.create<AffineForOp>(loc, 0, D_J/MM_J);
  Value j0Var = j0LoopOp.getInductionVar();
  builder.setInsertionPoint(&j0LoopOp.region().front(),
			    j0LoopOp.region().front().begin());

  Value bt;
  if (PACK_TRANSPOSE) {

    int64_t btShape[2];
    btShape[0] = MM_J;
    btShape[1] = MM_K;
    MemRefType btType = MemRefType::get(btShape, eltType);
    
    memref::AllocOp allocBtOp = builder.create<memref::AllocOp>(loc, btType);
    bt = allocBtOp.memref();

    AffineForOp kLoopOp = builder.create<AffineForOp>(loc, 0, MM_K);
    Value kVar = kLoopOp.getInductionVar();
    builder.setInsertionPoint(&kLoopOp.region().front(),
			      kLoopOp.region().front().begin());

    AffineForOp jLoopOp = builder.create<AffineForOp>(loc, 0, MM_J);
    Value jVar = jLoopOp.getInductionVar();
    builder.setInsertionPoint(&jLoopOp.region().front(),
			      jLoopOp.region().front().begin());

    auto kExpr = builder.getAffineDimExpr(0) + (builder.getAffineDimExpr(1) * MM_K);
    auto jExpr = builder.getAffineDimExpr(2) + (builder.getAffineDimExpr(3) * MM_J);
    AffineMap BAM = AffineMap::get(4, 0, {kExpr, jExpr}, builder.getContext());
    OperationState BLoadState(loc, AffineLoadOp::getOperationName());
    AffineLoadOp::build(builder, BLoadState, B,
			BAM, {kVar, k0Var, jVar, j0Var});
    Operation *BLoadOpPtr = builder.createOperation(BLoadState);
    AffineLoadOp BLoadOp = dyn_cast<AffineLoadOp>(BLoadOpPtr);
    Value elt = BLoadOp.result();

    auto identity0Expr = builder.getAffineDimExpr(0);
    auto identity1Expr = builder.getAffineDimExpr(1);
    AffineMap outputAM = AffineMap::get(2, 0,
					{identity0Expr, identity1Expr},
					builder.getContext());
    OperationState outputStoreState(loc,
				    AffineStoreOp::getOperationName());
    AffineStoreOp::build(builder, outputStoreState,
			 elt, bt, outputAM,
			 {jVar, kVar});
    builder.createOperation(outputStoreState);
    builder.setInsertionPointAfter(kLoopOp);
  }

  AffineForOp kLoopOp = builder.create<AffineForOp>(loc, 0, MM_K, MM_J);
  Value kVar = kLoopOp.getInductionVar();
  builder.setInsertionPoint(&kLoopOp.region().front(),
			    kLoopOp.region().front().begin());

  AffineForOp jLoopOp = builder.create<AffineForOp>(loc, 0, MM_J);
  Value jVar = jLoopOp.getInductionVar();
  builder.setInsertionPoint(&jLoopOp.region().front(),
			    jLoopOp.region().front().begin());

  auto identity0Expr = builder.getAffineDimExpr(0);
  Value bLoads[MM_J];
  if (PACK_TRANSPOSE) {
    for (int64_t i = 0; i < MM_J; i++) {
      auto incrExpr = builder.getAffineDimExpr(1) + i;
      AffineMap BAM = AffineMap::get(2, 0, {identity0Expr, incrExpr}, builder.getContext());
      OperationState BLoadState(loc, AffineLoadOp::getOperationName());
      AffineLoadOp::build(builder, BLoadState, bt,
			  BAM, {jVar, kVar});
      Operation *BLoadOpPtr = builder.createOperation(BLoadState);
      AffineLoadOp BLoadOp = dyn_cast<AffineLoadOp>(BLoadOpPtr);
      bLoads[i] = BLoadOp.result();
    }
  }
  else {
    auto identity1Expr = builder.getAffineDimExpr(1);
    for (int64_t i = 0; i < MM_J; i++) {
      auto incrExpr = builder.getAffineDimExpr(0) + i;
      AffineMap BAM = AffineMap::get(2, 0, {incrExpr, identity1Expr}, builder.getContext());
      OperationState BLoadState(loc, AffineLoadOp::getOperationName());
      AffineLoadOp::build(builder, BLoadState, B,
			  BAM, {kVar, jVar});
      Operation *BLoadOpPtr = builder.createOperation(BLoadState);
      AffineLoadOp BLoadOp = dyn_cast<AffineLoadOp>(BLoadOpPtr);
      bLoads[i] = BLoadOp.result();
    }
  }
  AffineForOp iLoopOp = builder.create<AffineForOp>(loc, 0, MM_I);
  Value iVar = iLoopOp.getInductionVar();
  builder.setInsertionPoint(&iLoopOp.region().front(),
			    iLoopOp.region().front().begin());

  auto jExpr = builder.getAffineDimExpr(1) + (builder.getAffineDimExpr(2) * MM_J) ;
  AffineMap CAM = AffineMap::get(3, 0, {identity0Expr, jExpr}, builder.getContext());
  OperationState CLoadState(loc, AffineLoadOp::getOperationName());
  AffineLoadOp::build(builder, CLoadState, C,
		      CAM, {iVar, jVar, j0Var});
  Operation *CLoadOpPtr = builder.createOperation(CLoadState);
  AffineLoadOp CLoadOp = dyn_cast<AffineLoadOp>(CLoadOpPtr);

  Value acc = CLoadOp.result();

  for (int64_t c = 0; c < MM_J; c++) {
    auto incrExpr = builder.getAffineDimExpr(1) + c;
    AffineMap AAM = AffineMap::get(2, 0, {identity0Expr, incrExpr}, builder.getContext());
    OperationState ALoadState(loc, AffineLoadOp::getOperationName());
    AffineLoadOp::build(builder, ALoadState, A,
			AAM, {iVar, kVar});
    Operation *ALoadOpPtr = builder.createOperation(ALoadState);
    AffineLoadOp ALoadOp = dyn_cast<AffineLoadOp>(ALoadOpPtr);
    Value aElt = ALoadOp.result();
    Value bElt = bLoads[c];
    MulFOp mulfOp = builder.create<MulFOp>(loc, aElt, bElt);
    AddFOp addfOp = builder.create<AddFOp>(loc, mulfOp.result(), acc);
    acc = addfOp.result();
  }

  OperationState outputStoreState(loc,
				  AffineStoreOp::getOperationName());
  AffineStoreOp::build(builder, outputStoreState,
		       acc, C, CAM, {iVar, jVar, j0Var});
  builder.createOperation(outputStoreState);
  matmulOp.erase();
}

void grabMatmulOps(std::list<linalg::MatmulOp> &matmulOps, Operation & op) {
  if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
    matmulOps.push_back(matmulOp);
  }
  else {
    for (Region &r: op.getRegions()) {
      for (Block &b: r.getBlocks()) {
	for (Operation &deeperOp: b.getOperations()) {
	  grabMatmulOps(matmulOps, deeperOp);
	}
      }
    }
  }
}

void lowerMatmuls(FuncOp f) {
  std::list<linalg::MatmulOp> matmulOps;
  for (auto &op : f.front()) {
    grabMatmulOps(matmulOps, op);
  }
  for (auto matmulOp: matmulOps) {
    lowerMatmul(matmulOp);
  }
}

void grabConvOps(std::list<linalg::ConvInputNHWCFilterHWCFOp> &convOps, Operation &op) {
  if (auto convOp = dyn_cast<linalg::ConvInputNHWCFilterHWCFOp>(op)) {
      convOps.push_back(convOp);
  }
  else {
    for (Region &r: op.getRegions()) {
      for (Block &b: r.getBlocks()) {
	for (Operation &deeperOp: b.getOperations()) {
	  grabConvOps(convOps, deeperOp);
	}
      }
    }
  }
}

void lowerConvolutions(FuncOp f) {
  // Extract all the convolutions of the function in a list
  std::list<linalg::ConvInputNHWCFilterHWCFOp> convOps;
  for (auto &op : f.front()) {
    grabConvOps(convOps, op);
  }
  // Then 
  for (auto convOp: convOps) {

    OpBuilder builder(convOp);
    Location loc = convOp.getOperation()->getLoc();

    // Get data
    
    Value input = convOp.inputs()[0];
    Value kernel = convOp.inputs()[1];
    MemRefType kernelMemRefType = kernel.getType().cast<MemRefType>();
    ArrayRef<int64_t> kernelShape = kernelMemRefType.getShape();
    Value output = convOp.outputs()[0];

    // Check structure

    bool strideOne = true;
    int64_t SI1;
    int64_t SI2;
    unsigned i = 0;
    for (llvm::APInt s: convOp.strides()) {

      if (i == 1) SI1 = s.getLimitedValue();
      else if (i == 2) SI2 = s.getLimitedValue();
      
      if (s.getLimitedValue() != 1) {
	strideOne = false;
      }
      
      i++;
    }
    
    if (kernelShape[0] == 1 && kernelShape[1] == 1) {
      Attribute zeroAttr = IntegerAttr::get(builder.getIndexType(), 0);
      ConstantOp zeroConstantOp = builder.create<ConstantOp>(loc, zeroAttr);
      Value zeroConstant = zeroConstantOp.getResult();
      Value inputPrime;
      if (strideOne) {
	inputPrime = input;
      }
      else {
	inputPrime = packAt(builder, loc, SI1, SI2, zeroConstant, input);
      }

      baseCase(builder, loc, zeroConstant, inputPrime, kernel, output);

      if (!strideOne) {
	builder.setInsertionPointAfter(convOp);
	builder.create<memref::DeallocOp>(loc, inputPrime);
      }
      
      convOp.erase();
    }
  }
}

void PrimeLinalgToAffinePass::runOnFunction() {
  lowerConvolutions(getFunction());
  lowerMatmuls(getFunction());
}

namespace mlir {
  void registerPrimeLinalgToAffinePass() {
    PassRegistration<PrimeLinalgToAffinePass>(PASS_NAME,
					      "Convert convolution operations to affine loops in an optimize fashion.");
  }
} // namespace mlir
