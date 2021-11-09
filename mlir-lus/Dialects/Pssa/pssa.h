// lus/pssa.h - Pssa dialect definition -*- C++ -*- //

#ifndef PSSA_H
#define PSSA_H

#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir {
  namespace pssa {

    struct PssaInlinerInterface : public DialectInlinerInterface {
      using DialectInlinerInterface::DialectInlinerInterface;
      bool isLegalToInline(Operation *, Region *, bool,
			   BlockAndValueMapping &) const final {
	return true;
      }
    };

    class Pssa : public Dialect {
    public:
      
      static llvm::StringRef getDialectNamespace() { return "pssa"; }
      explicit Pssa(MLIRContext *context);
    };
  }
}

#endif
