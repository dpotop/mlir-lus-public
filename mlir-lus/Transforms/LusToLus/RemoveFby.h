// -*- C++ -*- //

#ifndef REMOVE_FBY_H
#define REMOVE_FBY_H

#include <functional>
#include <list>
#include "../../Dialects/Lus/Node.h"

using namespace std;

namespace mlir {
  namespace lus {

    struct RemoveFby : public unary_function < NodeOp, void > {
    public:
      void operator() (NodeOp nop);
    private:
      list<Operation*> former;
      void removeFby(Operation* op);
    };
    
  }
}

#endif
