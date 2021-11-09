#include "lus.h"
#include "FbyOp.h"
#include "PreOp.h"
#include "WhenOp.h"
#include "MergeOp.h"
#include "Node.h"
#include "Instance.h"

namespace mlir {
  namespace lus {

    
    Lus::Lus(MLIRContext *context) :
      Dialect(getDialectNamespace(),context,TypeID::get<Lus>()) {
      addTypes <NodeType,
		YieldType,
		WhenType> () ;
      addOperations <
	PreOp,
	WhenOp,
	MergeOp,
	NodeOp,
	YieldOp,
	FbyOp,
	InstanceOp >() ;
      addInterfaces<LusInlinerInterface>();
    }

    void Lus::printType(Type, DialectAsmPrinter &) const {

    }
  }
}
