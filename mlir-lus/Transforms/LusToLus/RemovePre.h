// -*- C++ -*- //

#ifndef REMOVE_PRE_H
#define REMOVE_PRE_H

#include <functional>
#include <list>
#include "../../Dialects/Lus/Node.h"

using namespace std;

namespace mlir {
  namespace lus {

    struct RemovePre : public unary_function < NodeOp, void > {
    public:
      void operator() (NodeOp nop);
    private:
      NodeOp nodeOp;
      list<Operation*> former;
      void removePre(Operation* op);
    };
    
  }
}

#endif
