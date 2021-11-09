// -*- C++ -*- //

#ifndef GEN_OUTPUTS_H
#define GEN_OUTPUTS_H

#include <functional>
#include <unordered_set>
#include "../../Dialects/Lus/Node.h"

using namespace std;

namespace mlir {
  namespace lus {
    struct GenOutputs : public unary_function <NodeOp, void > {
    public:
      void operator() (NodeOp nodeOp);
    };
  }
}

#endif
