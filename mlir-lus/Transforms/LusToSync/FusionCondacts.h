// -*- C++ -*- //

#ifndef FUSION_CONDACTS_H
#define FUSION_CONDACTS_H

#include <functional>
#include <list>
#include <unordered_set>
#include "../../Dialects/Pssa/CondactOp.h"
#include "../../Dialects/Lus/Node.h"

using namespace std;

namespace mlir {
  namespace pssa {
    
    class FusionCondacts : public unary_function < lus::NodeOp, void > {
    public:
      void operator() (lus::NodeOp nodeOp);
    private:

      /// Fusion when possible the condacts of the region
      void apply(Region& body);

      /// Fusion the two condacts
      CondactOp fusion(CondactOp prev, CondactOp curr);

      /// Check if condacts can fusion
      bool canFusion(CondactOp first, CondactOp second);
      
      /// In code outside of condact, replace values stored in a by
      /// the results of condact
      void replaceResultsOutside(ArrayRef<Value> a, CondactOp condact);

      /// In code inside of condact, replace results of fst condact by values
      /// yield in fst
      void replaceResultsInside(CondactOp fst, CondactOp condactOp);
    };
    
  }
}


#endif
