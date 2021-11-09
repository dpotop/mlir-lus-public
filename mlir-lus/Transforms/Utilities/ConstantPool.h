// -*- C++ -*- //

#ifndef CONSTANT_POOL_H
#define CONSTANT_POOL_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "ValueHash.h"
#include "../../Dialects/Sync/Node.h"
#include <unordered_map>
#include <vector>

namespace mlir {

  struct ConstantPool {
  public:
    ConstantPool(sync::NodeOp node): myBuilder(node.getBody()) {}
    ConstantPool(FuncOp func): myBuilder(func.getBody()) {}
    ConstantPool(Region *region): myBuilder(region) {}
    Value getZero(Type t);
    Value getOne(Type t);
    Value getBool(int b);
    Value getVector(std::vector<bool> data, Type t);
    Value getUndef(Type t);
    Value buildInt(Type t, long v);
    Value buildFuncPointer(FuncOp funcOp);
    Value negate(OpBuilder &builder, Value v);
    Value increment(OpBuilder &builder, Value v);
    Value castToBool(OpBuilder &builder, Value v);
    Value castToIndex(OpBuilder &builder, Value v);
    Value extract(OpBuilder &builder, Value vect, Value ind);
    Value buildDataSize(Type t);
    Value buildNumDims(Type t);
  private:
    OpBuilder myBuilder;
    std::unordered_map<Type, Value, TypeHash> zeros;
    std::unordered_map<Type, Value, TypeHash> ones;
    std::unordered_map<Value, long, ValueHash> intToBool;
    std::unordered_map< std::vector<bool>, Value > flags;
    std::unordered_map< std::vector<bool>, Value > vectors;
    std::unordered_map< Value, Value, ValueHash > counters;
  };
  
}

#endif
