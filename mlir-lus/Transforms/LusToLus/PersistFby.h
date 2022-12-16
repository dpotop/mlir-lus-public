// -*- C++ -*- //

#ifndef PERSIST_FBY_H
#define PERSIST_FBY_H

#include <functional>
#include <list>
#include "../../Dialects/Lus/Node.h"
#include "../../Dialects/Lus/ClockTree.h"

using namespace std;

namespace mlir {
  namespace lus {

    struct PersistFby : public unary_function < NodeOp, void > {
    public:
      void operator() (NodeOp nop);
    private:
      ClockTree* clockTree;
      Operation* nodeOp;
      void persistFby(Operation* op, Value valCond);
    };
    
  }
}

#endif
