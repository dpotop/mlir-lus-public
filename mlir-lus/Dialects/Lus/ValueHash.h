// -*- C++ -*- //

#ifndef MLIRLUS_AUX_H
#define MLIRLUS_AUX_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <vector>

namespace mlir {

  // This ValueHash is used in several places: ClockAnalysis.h.
  // CondToPred.h. EnsureDominance.h. Hence, it's better to
  // extract it here.
  struct ValueHash {
    template <class Value>
    std::size_t operator() (const Value& value) const {
      // IR/Value.h
      //::llvm::hash_code hash_value(Value value) can be used
      return hash_value(value);
      // return std::hash<void*>()(value.getAsOpaquePointer());
    }
  };

}

#endif
