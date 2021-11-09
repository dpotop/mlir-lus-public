#include "OperationsAux.h"

namespace mlir {

  ConstantOp OperationsAux::buildBoolVector(OpBuilder &builder,
					    std::vector<bool> data,
					    Type t) {
    std::vector<int64_t> shape;
    shape.push_back(data.size());
    VectorType vectorType = VectorType::get(shape, t);
    std::vector<Attribute> dataAttributes;
    for (bool b : data) {
      Attribute attr = IntegerAttr::get(t, b);
      dataAttributes.push_back(attr);
    }
    Attribute vectorAttr = DenseElementsAttr::get(vectorType,
						  dataAttributes);
    OperationState state(builder.getUnknownLoc(),
			 ConstantOp::getOperationName());
    ConstantOp::build(builder, state, vectorAttr);
    Operation* constantOpPtr = builder.createOperation(state);
    ConstantOp constantOp = dyn_cast<ConstantOp>(constantOpPtr);
    return constantOp;
  }

}
