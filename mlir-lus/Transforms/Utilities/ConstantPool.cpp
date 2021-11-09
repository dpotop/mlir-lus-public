#include "ConstantPool.h"
#include "../../Dialects/Sync/UndefOp.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"

namespace mlir {

  
  Value ConstantPool::getZero(Type t) {
    Value zero;
    if (zeros.count(t) == 0) {
      zero = buildInt(t, 0);
    }
    else {
      zero = zeros[t];
    }
    return zero;
  }

  Value ConstantPool::getOne(Type t) {
    Value one;
    if (ones.count(t) == 0) {
      one = buildInt(t, 1);
    }
    else {
      one = ones[t];
    }
    return one;
  }

  Value ConstantPool::getBool(int b) {
    if (b) {
      return getOne(myBuilder.getI1Type());
    }
    else {
      return getZero(myBuilder.getI1Type());
    }
  }

  Value ConstantPool::getVector(std::vector<bool> data, Type t) {
    std::vector<int64_t> shape(data.size());
    VectorType vectType = VectorType::get(shape, t);
    std::vector<Attribute> dataAttrs;
    for (bool b : data) {
      Attribute attr = IntegerAttr::get(t, b);
      dataAttrs.push_back(attr);
    }
    Attribute vectorAttr = DenseElementsAttr::get(vectType, dataAttrs);
    OperationState state(myBuilder.getUnknownLoc(),
			 ConstantOp::getOperationName());
    ConstantOp::build(myBuilder, state, vectorAttr);
    Operation* constantOpPtr = myBuilder.createOperation(state);
    ConstantOp constantOp = dyn_cast<ConstantOp>(constantOpPtr);
    return constantOp.getResult();
  }

  Value ConstantPool::getUndef(Type t) {
    OperationState undefState(myBuilder.getUnknownLoc(),
			      sync::UndefOp::getOperationName());
    sync::UndefOp::build(myBuilder, undefState, t);
    Operation *undefOpPtr = myBuilder.createOperation(undefState);
    sync::UndefOp undefOp = dyn_cast<sync::UndefOp>(undefOpPtr);
    return undefOp.getResult();
  }
  
  Value ConstantPool::buildInt(Type type, long value) {
    Attribute attr = IntegerAttr::get(type, value);
    OperationState state(myBuilder.getUnknownLoc(),
			 ConstantOp::getOperationName());
    ConstantOp::build(myBuilder, state, attr);
    Operation* constantOpPtr = myBuilder.createOperation(state);
    ConstantOp constantOp = dyn_cast<ConstantOp>(constantOpPtr);
    Value v = constantOp.getResult();
    return v;
  }

  Value ConstantPool::buildFuncPointer(FuncOp funcOp) {
    Attribute nameAttr = FlatSymbolRefAttr::get(myBuilder.getContext(),
						funcOp.sym_name());
    OperationState constantState(myBuilder.getUnknownLoc(),
				 ConstantOp::getOperationName());
    ConstantOp::build(myBuilder, constantState, funcOp.getType(), nameAttr);
    Operation *constantOpPtr = myBuilder.createOperation(constantState);
    ConstantOp constantOp = dyn_cast<ConstantOp>(constantOpPtr);
    return constantOp.getResult();
  }

  Value ConstantPool::negate(OpBuilder &builder, Value v) {
    Value boolValue = castToBool(builder, v);
    Value one = getOne(myBuilder.getI1Type());
    Value zero = getZero(myBuilder.getI1Type());
    OperationState state(boolValue.getLoc(), SelectOp::getOperationName());
    SelectOp::build(builder, state, boolValue, zero, one);
    Operation * selectOpPtr = builder.createOperation(state);
    SelectOp selectOp = dyn_cast<SelectOp>(selectOpPtr);
    Value res = selectOp.getResult();
    return res;
  }

  Value ConstantPool::increment(OpBuilder &builder, Value v) {
    Value one = getOne(v.getType());
    OperationState addState(builder.getUnknownLoc(),
			    AddIOp::getOperationName());
    AddIOp::build(builder, addState, v, one);
    Operation* addOpPtr = builder.createOperation(addState);
    AddIOp addOp = dyn_cast<AddIOp>(addOpPtr);
    return addOp.getResult();
  }

  Value ConstantPool::castToBool(OpBuilder& builder, Value v) {
    if (v.getType() == myBuilder.getI1Type()) {
      return v;
    }
    Value zero = getZero(v.getType());
    OperationState state(v.getLoc(), CmpIOp::getOperationName());
    CmpIOp::build(builder, state, builder.getI1Type(),
		  CmpIPredicate::ne, v, zero);
    Operation *cmpOpPtr = builder.createOperation(state);
    CmpIOp cmpOp = dyn_cast<CmpIOp>(cmpOpPtr);
    return cmpOp.getResult();
  }

  Value ConstantPool::castToIndex(OpBuilder &builder, Value v) {
    if (v.getType() == myBuilder.getIndexType()) {
      return v;
    }
    else if (v.getType() == myBuilder.getI1Type()) {
      Value one = getOne(myBuilder.getIndexType());
      Value zero = getZero(myBuilder.getIndexType());
      OperationState selectState(builder.getUnknownLoc(),
				 SelectOp::getOperationName());
      SelectOp::build(builder, selectState, v, one, zero);
      Operation *selectOpPtr = builder.createOperation(selectState);
      SelectOp selectOp = dyn_cast<SelectOp>(selectOpPtr);
      return selectOp.getResult();
    }
    else {
      OperationState castState(builder.getUnknownLoc(),
			       IndexCastOp::getOperationName());
      IndexCastOp::build(builder, castState, builder.getIndexType(), v);
      Operation *indexCastOpPtr = builder.createOperation(castState);
      IndexCastOp indexCastOp = dyn_cast<IndexCastOp>(indexCastOpPtr);
      return indexCastOp.getResult();
    }
  }

  Value ConstantPool::extract(OpBuilder &builder, Value vect, Value ind) {
    OperationState state(builder.getUnknownLoc(),
			 vector::ExtractElementOp::getOperationName());
    Type elementType = vect.getType().cast<VectorType>().getElementType();
    vector::ExtractElementOp::build(builder, state, elementType, vect, ind);
    Operation *extractPtr = builder.createOperation(state);
    vector::ExtractElementOp extract(extractPtr);
    return extract.getResult();
  }

  Value ConstantPool::buildDataSize(Type t) {
    Type elt;
    if (t.isa<ShapedType>()) {
      ShapedType st = t.cast<ShapedType>();
      elt = st.getElementType();
    }
    else {
      elt = t;
    }
    if (elt == myBuilder.getI32Type() || elt == myBuilder.getF32Type()) {
      return buildInt(myBuilder.getI32Type(), 4);
    }
    else if (elt == myBuilder.getI64Type() || elt == myBuilder.getF64Type()) {
      return buildInt(myBuilder.getI32Type(), 8);
    }
    else if (elt == myBuilder.getI1Type()) {
      return buildInt(myBuilder.getI32Type(), 1);
    }
    else if (elt == myBuilder.getIntegerType(8)) {
      return buildInt(myBuilder.getI32Type(), 1);
    }
    else {
      assert(false);
    }
  }

  Value ConstantPool::buildNumDims(Type t) {
    if (t.isa<ShapedType>()) {
      ShapedType st = t.cast<ShapedType>();
      return buildInt(myBuilder.getI32Type(),
		      st.getShape().size());
    }
    else {
      assert(false);
    }
  }
  
}
