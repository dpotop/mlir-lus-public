// -*- C++ -*- //

#ifndef ENSURE_DOMINANCE_H
#define ENSURE_DOMINANCE_H

#include <functional>
#include <list>
#include <unordered_set>
#include "../../Dialects/Lus/Node.h"
#include "../Utilities/ValueHash.h"
#include "../../Dialects/Lus/ClockTree.h"

using namespace std;

namespace mlir {
  namespace lus {

    struct EnsureDominance : public unary_function < NodeOp, void > {
    public:
      void operator() (NodeOp nodeOp);
    private:
      ClockTree *clockTree;

      /// The algorithm iterates on these unordered ops and picks the next one
      /// which must be sorted
      list<Operation*> unorderedOps;

      /// The algorithm stores sorted ops here (in the right order)
      list<Operation*> orderedOps;

      /// The values defined at this point of ordered ops (used to decide
      /// if an unordered op can be added regarding to dominance rules)
      unordered_set<Value, ValueHash> definedValues;

      /// Check if op can be added to ordered ops without violating dominance
      bool dominanceCorrection(Operation* op);

      /// Get the next unordered op which is on the same clock than the last
      /// ordered op's clock (returns false if doesn't exist) ; remove it from
      /// the unordered ops, add it to the ordered
      bool nextOnSameClock();

      /// Get the next unordered op which is on a subclock of the last
      /// ordered op's clock (returns false if doesn't exist) ; remove it from
      /// the unordered ops, add it to the ordered ops
      bool nextOnSubClock();

      /// Get the next unordered op whose clock is as tiny as possible ;
      /// remove it from the unordered ops, add it to the ordered ops
      bool nextOnTiniestClock();

      /// Sort unorderedOps in orderedOps
      void order();
    };
  }
}

#endif
