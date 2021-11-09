#include "Sync.h"
#include "mlir/IR/Builders.h"
#include "SignalTypes.h"
#include "NodeType.h"
#include "Node.h"
#include "HaltOp.h"
#include "InputOp.h"
#include "OutputOp.h"
#include "TickOp.h"
#include "SyncOp.h"
#include "InOp.h"
#include "OutOp.h"
#include "InstOp.h"
#include "UndefOp.h"
#include "SelectOp.h"

namespace mlir {
  namespace sync {

    
    Sync::Sync(MLIRContext *context) :
      Dialect(getDialectNamespace(),context,TypeID::get<Sync>()) {
      addTypes <SiginType, SigoutType, EventType, NodeType> () ;
      addOperations <NodeOp, InstOp, HaltOp,
		     TickOp, SyncOp, UndefOp, SelectOp,
		     InputOp, OutputOp, InOp, OutOp> ();
      addInterfaces<SyncInlinerInterface>();
    }

    Type Sync::parseType(DialectAsmParser &parser) const {
      Type t;
      if (succeeded(parser.parseOptionalKeyword("sigin"))
	  && succeeded(parser.parseLess())
	  && succeeded(parser.parseType(t))
	  && succeeded(parser.parseGreater())) {
	return SiginType::get(parser.getBuilder().getContext(), t);
      }
      if (succeeded(parser.parseOptionalKeyword("sigout"))
	  && succeeded(parser.parseLess())
	  && succeeded(parser.parseType(t))
	  && succeeded(parser.parseGreater())) {
      	return SigoutType::get(parser.getBuilder().getContext(), t);
      }
      if (succeeded(parser.parseOptionalKeyword("event")))
	return EventType::get(parser.getBuilder().getContext());
      return Type();
    }

    void Sync::printType(Type type, DialectAsmPrinter &printer) const {
      if (type.isa<SiginType>()) {
	SiginType st = type.cast<SiginType>();
	printer << "sigin<" << st.getType() << ">";
      }
      else if (type.isa<SigoutType>()) {
	SigoutType st = type.cast<SigoutType>();
	printer << "sigout<" << st.getType() << ">";
      }
      else if (type.isa<EventType>()) {
	printer << "event";
      }
    }
  }
}
