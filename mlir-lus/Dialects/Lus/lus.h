// lus/lus.h - Lus dialect definition -*- C++ -*- //

#ifndef MLIRLUS_H
#define MLIRLUS_H

#include "mlir/Transforms/InliningUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir {
  namespace lus {

    struct LusInlinerInterface : public DialectInlinerInterface {
      using DialectInlinerInterface::DialectInlinerInterface;
      bool isLegalToInline(Operation *, Region *, bool,
			   BlockAndValueMapping &) const final {
	return true;
      }
    };

    class Lus : public Dialect {
    public:
      static llvm::StringRef getDialectNamespace() { return "lus"; }
      explicit Lus(MLIRContext *context) ;
      void printType(Type, DialectAsmPrinter &) const;
    };
  }
}

#endif
