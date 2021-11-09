// Sync dialect definition -*- C++ -*- //

#ifndef DIALECT_SYNC_H
#define DIALECT_SYNC_H

#include "mlir/Transforms/InliningUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir {
  namespace sync {

    struct SyncInlinerInterface : public DialectInlinerInterface {
      using DialectInlinerInterface::DialectInlinerInterface;
      bool isLegalToInline(Operation *, Region *, bool,
			   BlockAndValueMapping &) const final {
	return true;
      }
    };

    class EventType : public Type::TypeBase<EventType, Type, TypeStorage> {
    public:
      using Base::Base;
      static EventType get(MLIRContext *context) {
	return Base::get(context) ;
      }
      unsigned getWidth() { return 1 ; }
    };
    
    class Sync : public Dialect {
    public:
      static llvm::StringRef getDialectNamespace() { return "sync"; }
      explicit Sync(MLIRContext *context) ;

      Type parseType(DialectAsmParser &parser) const override;
      void printType(Type type, DialectAsmPrinter &printer) const override;
    };
  }
}

#endif
