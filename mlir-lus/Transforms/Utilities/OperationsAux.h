// -*- C++ -*- //

#ifndef OPERATIONS_AUX_H
#define OPERATIONS_AUX_H

#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "ConstantPool.h"

namespace mlir {
  struct OperationsAux {
    static ConstantOp buildBoolVector(OpBuilder &builder,
				      std::vector<bool> data,
				      Type t);
  };
}  

#endif
