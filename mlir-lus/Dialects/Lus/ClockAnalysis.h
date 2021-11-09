// -*- C++ -*- //

#ifndef CLOCK_ANALYSIS_NEW_H
#define CLOCK_ANALYSIS_NEW_H

#include <list>
#include <unordered_map>
#include <unordered_set>
#include "Clock.h"
#include "ClockTree.h"
#include "ValueHash.h"

// #include <optional>
// Needed for struct ValueHash - I could move it here upon cleaning
// #include "LusHelpers.h"
// Needed to include the RegionKindInterface and its trait
// #include "mlir/IR/RegionKindInterface.h"
// Needed for the base node

using namespace std;

namespace mlir {
  namespace lus {
   
    // This class groups and encapsulates the data structures
    // and access methods to clocks used during the analysis of
    // one node.
    class ClockRepository {
      //-------------------------------------------------------
      // Map from MLIR SSA values to clocks. This is the data
      // structure on which clock calculus (unification) is
      // performed.
      unordered_map<Value, Clock*, ValueHash> clocksOnValues;
      // List of all the clock objects that were ever created in this
      // ClockAnalysis instance. It is only used at destruction time,
      // to make sure that even clocks discarded during unification are
      // deleted (a form of garbage collection). No need therefore to
      // use a smarter data structure.
      list<Clock*> clocksDatapool;
      // The base clock of the node
      BaseClock baseClock  ;
    public:
      //-------------------------------------------------------
      // Accessor - only update possible is through substitution
      Clock& getClock(Value v) {
	Clock* r = clocksOnValues[v] ;
	assert(r!= NULL) ;
	return *r ;
      }
      void substitute(FreeClock& clock1, Clock& clock2) ;
      // Substitute all remaining free clocks with the base clock
      void substituteBaseClock() ;
      
      //-------------------------------------------------------
      // Constructor and destructor
      ClockRepository() : clocksOnValues(), clocksDatapool(), baseClock() {}
      ~ClockRepository() {
	for (auto c : clocksDatapool) { delete c; }
	clocksOnValues.clear();
	clocksDatapool.clear();
      }      
      void printClocksOnValues(raw_ostream&err,int indent_level) ;
      
      //-------------------------------------------------------
      // Building clocks
      //
      // Assign a new FreeClock to an SSA value v. This only happens
      // at init time.
      void initializeClock(Value& v) {
	FreeClock* r = new FreeClock();
	clocksDatapool.push_back(r);
	clocksOnValues.insert(pair<Value,Clock*>(v, r));
      }
      // Build a new WhenClock of base cl and condition cond
      WhenClock& buildWhenClock(Clock& cl, Cond<Value> cond) {
	auto r = new WhenClock(cl, cond);
	clocksDatapool.push_back(r);
	return *r;
      }
      // Build the complementary of a WhenClock w.r.t. its base
      // clock.
      WhenClock& buildComplementClock(WhenClock& whenClk) {
	const Cond<Value>* cond = whenClk.cond().buildComplement() ;
	return buildWhenClock(whenClk.clock(),*cond) ;
      }

      
      
    } ;


    struct UnifyResult {
      FreeClock* toSubstitute ;
      Clock* substitute ;


      static UnifyResult Fail() {
	UnifyResult u ;
	u.toSubstitute = NULL ;
	u.substitute = NULL ;
	return u ;
      }
      static UnifyResult Success(FreeClock&toSubstituteIn,
				 Clock&substituteIn) {
	UnifyResult u ;
	u.toSubstitute = &toSubstituteIn ;
	u.substitute = &substituteIn ;
	return u ;
      }
      bool operator==(const UnifyResult&s) const {
	return ((*toSubstitute)==(*s.toSubstitute))
	  &&((*substitute)==(*s.substitute)) ;
      }
      bool isSuccess() const { return toSubstitute != NULL ; }
      FreeClock& getToSubstitute() {
	assert(isSuccess()) ;
	return *toSubstitute ;
      }
      Clock& getSubstitute() {
	assert(isSuccess()) ;
	return *substitute ;
      }      

      
    } ;

    
    class NodeOp ;

    class ClockAnalysis {
      // Clock analysis is performed on a per-node basis (one
      // instance of the class is created for each node).
      NodeOp &nodeOp;

      //
      ClockRepository clockRepository ;
      unordered_set<Clock*> neverUnify;
      
    public:
      // Public constructor and destructor
      ClockAnalysis(NodeOp& node)
	: nodeOp(node), clockRepository() {} 
      ~ClockAnalysis() {}

    private:

      bool analyseDone = false;
      bool treeBuilt = false;
      ClockTree tree;
      void buildClockTree(Operation* o);
      // The following routines are used in the implementation
      // of clock analysis.


      // Given two clocks, determine if they are compatible.
      // - If they are not, the success field of the return
      //   struct is false, and the two pointers are NULL.
      // - If they are, then I obtain the clock substitution I
      //   have to apply to effectively enforce unification.
      // With more complex clock calculi, there may be more than
      // one substitution to perform. Here, it's just the
      // replacement of one FreeClock with another clock.      
      UnifyResult unify(Clock& clock1, Clock& clock2);
      // Functions called by unify - description provided with
      // their implementation.
      UnifyResult unifyWhen(Clock& clock1, Clock& clock2);
      UnifyResult unifyFree(Clock& clock1, Clock& clock2);
      //
      void unificationError(Operation&,Value&,Value&,Clock&,Clock&) ;
      //
      LogicalResult synchronizeList(list<Value> valueList);
      //
      void initializeClocks(Operation* op);
      
      LogicalResult whenAnalyse(Operation*);
      LogicalResult mergeAnalyse(Operation*);
      LogicalResult yieldAnalyse(Operation*);
      LogicalResult preAnalyse(Operation*);
      LogicalResult simpleAnalyse(Operation*);
      
      LogicalResult analyseOp(Operation* op);

    public:
      // Compute all clocks.
      LogicalResult analyse(); 
      //
      bool enforceClockEquality(Clock&v1,Clock&v2) ;
      //
      ClockTree& getClockTree();

    };
  }
}

#endif
