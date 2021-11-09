// -*- C++ -*- //

#ifndef BUFFER_GENERATOR_H
#define BUFFER_GENERATOR_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "ConstantPool.h"

namespace mlir {
  struct BufferGenerator {
    static FunctionType inputFunction(OpBuilder &builder, Type t);
    static FunctionType outputFunction(OpBuilder &builder, Type t);
    static MemRefType abstractBufferizedType(Type t);
    static MemRefType concreteBufferizedType(Type t);
    static Value bufferize(OpBuilder &builder, Location loc,
			   ConstantPool &cp, Value v);
    static Value unbufferize(OpBuilder &, Location, ConstantPool &,
			     Type, Value);
    static Value allocateType(OpBuilder &builder, Location loc, Type t);
  };
}

#endif
