// lus/GenPredicates.h - GenPredicates class definition -*- C++ -*- //

#ifndef GEN_PREDICATES_H
#define GEN_PREDICATES_H

#include <functional>
#include "../../Dialects/Lus/Node.h"
#include "../../Dialects/Lus/WhenOp.h"
#include "../../Dialects/Lus/MergeOp.h"
#include "../../Dialects/Pssa/CreatePredOp.h"
#include "../Utilities/CondToPred.h"

using namespace std;

namespace mlir {
  namespace lus {
    
    class GenPredicates : public unary_function < NodeOp, void > {
    public:
      void operator() (NodeOp);
    private:

      /// The current node
      NodeOp nodeOp;

      /// Conditions (data & kperiodic) with the corresponding predicates
      CondToPred result;

      /// If needed, build and store predicate corresponding to when
      void handle(WhenOp);
      
      /// If needed, build and store predicates corresponding to merge
      /// (condition and complement)
      void handle(MergeOp);

      /// Stores predicate corresponding to data
      void handle(pssa::CreatePredOp);

      /// Gen predicates which are not op-dependant
      void finalize();

      /// If needed build and store predicate corresponding to value
      void buildCreatePredOp(Value);

      /// If needed build and store predicate corresponding to k-periodic word
      void buildCreatePredOp(KPeriodic);
    };
  }
}

#endif
