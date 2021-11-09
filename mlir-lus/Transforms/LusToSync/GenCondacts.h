// lus/LowerReactive.h - LowerReactive class definition -*- C++ -*- //

#ifndef GEN_CONDACTS_H
#define GEN_CONDACTS_H

#include <functional>
#include <unordered_set>
#include <map>
#include <list>
#include "../../Dialects/Lus/ClockTree.h"
#include "../../Dialects/Lus/Node.h"
#include "../Utilities/CondToPred.h"

using namespace std;

namespace mlir {
  namespace lus {

    struct GenCondacts : public unary_function <NodeOp, void > {
    public:

      void operator() (NodeOp nodeOp);

    private:
      
      unordered_set<Operation*> dishes;
      CondToPred condToPred;
      ClockTree* clockTree;
      
      /// Parse predicates in the node
      void storePredicates(Operation *op);
      
      /// Check if op should be inserted in a condact
      bool shouldBeCondacted(Operation *op);

      /// Predicate op
      Operation* insertInCondacts(Operation* op);

      /// Notify environment that newOp is condacted op
      void updateEnvironment(Operation* op, Operation *newOp);

      // Lower op if it is merge or when
      void removeReactive(Operation* op);
    };
  }
}

#endif
