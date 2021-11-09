#include "BufferGenerator.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {

  MemRefType BufferGenerator::concreteBufferizedType(Type t) {
    if (t.isa<ShapedType>()) {
      assert(t.isa<ShapedType>());
      ShapedType tt = t.cast<ShapedType>();
      MemRefType mrt = MemRefType::get(tt.getShape(), tt.getElementType());
      return mrt;
    }
    else {
      std::vector<int64_t> shape(1);
      shape[0] = 1;
      MemRefType mrt = MemRefType::get(shape, t);
      return mrt;
    }
  }

  MemRefType BufferGenerator::abstractBufferizedType(Type t) {
    // return concreteBufferizedType(t);
    if (t.isa<ShapedType>()) {
      assert(t.isa<ShapedType>());
      ShapedType tt = t.cast<ShapedType>();
      std::vector<int64_t> shape = tt.getShape();
      for (unsigned i = 0; i < shape.size(); i++) {
    	shape[i] = -1;
      }
      MemRefType mrt = MemRefType::get(shape, tt.getElementType());
      return mrt;
    }
    else {
      std::vector<int64_t> shape(1);
      shape[0] = -1;
      MemRefType mrt = MemRefType::get(shape, t);
      return mrt;
    }
  }
  
  Value BufferGenerator::bufferize(OpBuilder &builder, Location loc,
				   ConstantPool &cp,
				   Value v) {
    Type t = v.getType();
    MemRefType concBufTy = concreteBufferizedType(t);
    MemRefType absBufTy = abstractBufferizedType(t);
    if (t.isa<TensorType>()) {
      OperationState tToMState(loc, memref::BufferCastOp::getOperationName());
      memref::BufferCastOp::build(builder,tToMState, concBufTy, v);
      Operation* tToMOpPtr = builder.createOperation(tToMState);
      memref::BufferCastOp tToMOp = dyn_cast<memref::BufferCastOp>(tToMOpPtr);
      Value memRef = tToMOp.getResult();

      OperationState castState(loc, memref::CastOp::getOperationName());
      memref::CastOp::build(builder, castState, memRef, absBufTy);
      Operation *castOpPtr = builder.createOperation(castState);
      memref::CastOp castOp = dyn_cast<memref::CastOp>(castOpPtr);
      
      return castOp.getResult();
    }
    else {
      Value memRef = allocateType(builder, loc, t);
      Value zero = cp.getZero(builder.getIndexType());
      OperationState storeState(loc, memref::StoreOp::getOperationName());
      memref::StoreOp::build(builder, storeState, v, memRef, zero);
      builder.createOperation(storeState);
      return memRef;
    }
  }

  Value BufferGenerator::unbufferize(OpBuilder &builder, Location loc,
				     ConstantPool &cp,
				     Type initT, Value v) {
    if (initT.isa<TensorType>()) {
      OperationState tensorLoadState(loc,
				     memref::TensorLoadOp::getOperationName());
      memref::TensorLoadOp::build(builder, tensorLoadState, v);
      Operation *tensorLoadPtr = builder.createOperation(tensorLoadState);
      memref::TensorLoadOp tensorLoad = dyn_cast<memref::TensorLoadOp>(tensorLoadPtr);
      Value tensor = tensorLoad.getResult();
      OperationState tensorCastState(loc, tensor::CastOp::getOperationName());
      tensor::CastOp::build(builder, tensorCastState, initT, tensor);
      Operation *tCastOpPtr = builder.createOperation(tensorCastState);
      tensor::CastOp tCastOp = dyn_cast<tensor::CastOp>(tCastOpPtr);
      return tCastOp.getResult();
    }

    else if (!initT.isa<ShapedType>()) {
      Value zero = cp.getZero(builder.getIndexType());
      OperationState loadState(loc, memref::LoadOp::getOperationName());
      memref::LoadOp::build(builder, loadState, v, zero);
      Operation *loadOpPtr = builder.createOperation(loadState);
      memref::LoadOp loadOp = dyn_cast<memref::LoadOp>(loadOpPtr);
      return loadOp.getResult();
    }

    else {
      assert(false);
    }
  }
  
  Value BufferGenerator::allocateType(OpBuilder &builder, Location loc,
				      Type t) {

    MemRefType concBufTy = concreteBufferizedType(t);
    MemRefType absBufTy = abstractBufferizedType(t);
    
    OperationState allocState(loc, memref::AllocOp::getOperationName());
    memref::AllocOp::build(builder, allocState, concBufTy);
    Operation* allocOpPtr = builder.createOperation(allocState);
    memref::AllocOp allocOp = dyn_cast<memref::AllocOp>(allocOpPtr);
    Value memRef = allocOp.getResult();

    OperationState castState(loc, memref::CastOp::getOperationName());
    memref::CastOp::build(builder, castState, memRef, absBufTy);
    Operation *castOpPtr = builder.createOperation(castState);
    memref::CastOp castOp = dyn_cast<memref::CastOp>(castOpPtr);
    
    return castOp.getResult();
  }

  FunctionType BufferGenerator::inputFunction(OpBuilder &builder, Type t) {
    Type i32 = builder.getI32Type();
    MemRefType mrt = abstractBufferizedType(t);
    return builder.getFunctionType({i32, mrt}, {});
  }

  FunctionType BufferGenerator::outputFunction(OpBuilder &builder, Type t) {
    Type i32 = builder.getI32Type();
    MemRefType mrt = abstractBufferizedType(t);
    return builder.getFunctionType({i32, mrt}, {i32});
  }
}
