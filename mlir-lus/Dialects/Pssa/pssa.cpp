#include "pssa.h"
#include "CondactOp.h"
#include "CreatePredOp.h"
#include "OutputOp.h"

namespace mlir {
  namespace pssa {

    Pssa::Pssa(MLIRContext *context) :
	Dialect(getDialectNamespace(),context,TypeID::get<Pssa>()) {
      addOperations<YieldOp, CondactOp, CreatePredOp, OutputOp>() ;
	addInterfaces<PssaInlinerInterface>();
	addTypes<CreatePredType>();
      }
  }
}
