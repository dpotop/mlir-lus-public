#include "../../Tools/CommandLine.h"
#include "TestCondition.h"
#include "ClockAnalysis.h"

#include "../Pssa/pssa.h"
#include "Node.h"
#include "Yield.h"
#include "PreOp.h"
#include "WhenOp.h"
#include "MergeOp.h"
#include "../Sync/SyncOp.h"
#include "../Sync/TickOp.h"

using namespace std;

namespace mlir {
  namespace lus {
    bool ClockAnalysis::enforceClockEquality(Clock&clock1,
					     Clock&clock2) {
      if (neverUnify.count(&clock1) || neverUnify.count(&clock2)) {
	return true;
      }
      llvm::raw_fd_ostream err(2,false,true) ;
      if(verboseLevel>2) err << "    >>enforceClockEquality(begin)\n" ;
      if(clock1 != clock2) {
	if(verboseLevel>2) err << "    >>enforceClockEquality(0) - unification is needed\n" ;
	// Unification is needed. Attempt to do it.
	UnifyResult res = unify(clock1,clock2) ;
	if(res.isSuccess()) {
	  // Unification succeeded, perform substitution
	  if(verboseLevel>2) err << "    >>enforceClockEquality(1) - perform sub\n" ;
	  clockRepository.substitute(res.getToSubstitute(),res.getSubstitute()) ;
	  if(verboseLevel>2) {
	    err << "    >>enforceClockEquality(2) - resulting clocks:\n" ;
	    clockRepository.printClocksOnValues(err,2) ;
	  }
	} else {	  
	  if(verboseLevel>2) err << "    >>enforceClockEquality(3) - unification failure. Return.\n" ;
	  return false ;
	}
      } 
      if(verboseLevel>2) err << "    >>enforceClockEquality(end)\n" ;
      return true ;
    }

    void ClockRepository::printClocksOnValues(raw_ostream&os,int indent_level) {
      for (pair<Value, Clock*> p : clocksOnValues) {
	// Get the value and clock of the assignment
	Value v = get<0>(p);
	Clock* c = get<1>(p);
	assert(c != NULL) ;
	// Start printing
	for(int i=0;i<indent_level;i++)
	  os << "\t" ;
	debugPrintObj<Value>(v,os) ;
	os << "\t->" ;
	c->debugPrint(os) ;
	os << "\n" ;
      }
    }

    void ClockRepository::substitute(FreeClock& clock1, Clock& clock2) {
      // For debug printing
      llvm::raw_fd_ostream err(2,false,true) ;
      if(verboseLevel>2) {
	err << "\t===============================================================\n" ;
	err << "\tClockAnalysis::substitute " ;
      	clock1.debugPrint(err) ;
	err << " with " ;
	clock2.debugPrint(err) ;
	err << " in context:\n" ;
	printClocksOnValues(err,2) ;
      }
      
      // Traverse clocksOnValues
      for (pair<Value, Clock*> p : clocksOnValues) {
	Value v = get<0>(p);
	if(verboseLevel>2) {
	  err << "\t---------------------------------------------------------------\n" ;
	  err << "\tClockAnalysis::substitute on var " ;
	  debugPrintObj<Value>(v,err) ;
	  err << "\n" ;
	}
	if(verboseLevel>2) {
	  err << "\tTest clock1 has not been deleted(0): " ; clock1.debugPrint(err) ; err << "OK.\n" ;
	}	
	Clock* c = get<1>(p);
	assert(c != NULL) ;
	if ((*c) == clock1) {
	  // In this case, simply replace c with clock2 in clocksOnValues
	  if(verboseLevel>2) {
	    err << "\t\tclocksOnValues[" ; debugPrintObj<Value>(v,err) ;
	    err << "] := " ; clock2.debugPrint(err) ; err << "\n" ;
	  }
	  clocksOnValues[v] = &clock2;
	  if(verboseLevel>2) {
	    err << "\tTest clock1 has not been deleted(1): " ; clock1.debugPrint(err) ; err << "OK.\n" ;
	  }

	} else {
	  if(isa<WhenClock>(c)) {
	    // Even if it's not identical, if this is a WhenClock,
	    // I still have to traverse it to determine if clock1 is
	    // not used as a base clock.
	    WhenClock* wcn = cast<WhenClock>(c) ;
	    if(verboseLevel>2) {
	      err << "\t\t" ; wcn->debugPrint(err) ; err << "\n" ;
	      err << "\t\t" ; clock1.debugPrint(err) ; err << "\n" ;
	      err << "\t\t" ; clock2.debugPrint(err) ; err << "\n" ;
	    }
	    wcn->substitute(clock1,clock2) ;
	  }
	  if(verboseLevel>2) {
	    err << "\tTest clock1 has not been deleted(2): " ; clock1.debugPrint(err) ; err << "OK.\n" ;
	  }
	}
	if(verboseLevel>2) {
	  err << "\tClockAnalysis::substitute on var " ;
	  debugPrintObj<Value>(v,err) ;
	  err << " resulting clocks:\n" ;
	  printClocksOnValues(err,2) ;
	  err << "\n" ;
	}
      }
      if(verboseLevel>2) {
	err << "\tClockAnalysis<rk>::substitute(end)\n" ;
      }
    }
    // Substitute all remaining free clocks with the base clock
    void ClockRepository::substituteBaseClock() {
      for (pair<Value, Clock*> p : clocksOnValues) {
	Clock* c = std::get<1>(p);
	if(isa<FreeClock>(*c)) {
	  FreeClock& fclk = cast<FreeClock>(*c) ;
	  // Substitution is always possible (and necessary)
	  substitute(fclk,baseClock) ;
	}
      }
    }
       
    // Determining for two **different** clocks the least constrained
    // clock that complies with the requirements of both clocks. If the
    // clocks are equal, a useless substitution phase will be enforced.
    // - If the clocks are incompatible, the routine returns
    //   UnifyResult().
    // - If the clocks are compatible, the routine returns the
    //   substitution action that would make the clocks equal.
    //   (a single operation suffices in this clock calculus).
    UnifyResult ClockAnalysis::unify(Clock& clock1, Clock& clock2) {
      // For debug printing
      llvm::raw_fd_ostream err(2,false,true) ;
      if(verboseLevel>2) {
	err << "\tClockAnalysis::unify called on " ;
	clock1.debugPrint(err) ;
	err << " and " ;
	clock2.debugPrint(err) ;
	err << "\n" ;
      }   
      // Unification
      if (isa<WhenClock>(clock1)   && isa<WhenClock>(clock2)) {
	if(verboseLevel>2) err << "\tClockAnalysis::unify case 1\n" ;
	return unifyWhen(clock2,clock1) ;
      } else if(isa<FreeClock>(clock2)) {
	if(verboseLevel>2) err << "\tClockAnalysis::unify case 2\n" ;
	return unifyFree(clock2,clock1) ;
      } else if(isa<FreeClock>(clock1)) {
	if(verboseLevel>2) err << "\tClockAnalysis::unify case 3\n" ;
	return unifyFree(clock1,clock2) ;
      } else {
	if(verboseLevel>2) err << "\tClockAnalysis::unify case 4\n" ;
	err << "Clock analysis error:\n"
	    << "clock is neither Free, nor When - aborting...\n" ;
	assert(false) ;
      }
    }
    // Sub-function of unify that is called when both clocks are When clocks.
    UnifyResult ClockAnalysis::unifyWhen(Clock& clock1, Clock& clock2) {    
      WhenClock& wClk1 = cast<WhenClock>(clock1);
      WhenClock& wClk2 = cast<WhenClock>(clock2);
      if (wClk1.cond() != wClk2.cond()) {
	// Cannot unify condition clocks with different predicate.
	// Matching is syntactic, a condition is always a value,
	// which must be identical. This can be refined later.
	llvm::raw_fd_ostream err(2,false,true) ;
	err << "Clock analysis error:\n"
	    << "cannot unify When clocks on different conditions in:\n" ; 
	return UnifyResult::Fail() ;
      } else {
	if(verboseLevel>2) {
	  llvm::raw_fd_ostream err(2,false,true) ;
	  err << "\tClockAnalysis::unifyWhen case\n" ;
	}
	// I can unify one level. I need to see if I can unify - one
	// of the clocks is the unified one.
	return unify(wClk1.clock(), wClk2.clock()) ;
      }
    }
    // Sub-function of unify that is called when at least one of the clocks
    // is a FreeClock. It is assumed during the call that the first clock
    // is the FreeClock.
    UnifyResult ClockAnalysis::unifyFree(Clock& clock1, Clock& clock2) {
      FreeClock& ck = cast<FreeClock>(clock1) ;
      // Check whether clock2 is not a WhenClock based on ck, in which case
      // unification is impossible.
      if(isa<WhenClock>(clock2)) {
	WhenClock& wck = cast<WhenClock>(clock2) ;
	if(ck == wck.getBaseClock()) {
	  // Cannot unify a clock with one of its sub-clocks
	  llvm::raw_fd_ostream err(2,false,true) ;
	  err << "Clock analysis error:\n"
	      << "cannot unify clock with one of its sub-clocks in:\n" ; 
	  return UnifyResult::Fail() ;
	}
      }
      return UnifyResult::Success(ck,clock2) ;
    }
    void ClockAnalysis::unificationError(Operation&op,
					    Value&v1,
					    Value&v2,
					    Clock&clk1,
					    Clock&clk2) {
      // For debug printing
      llvm::raw_fd_ostream err(2,false,true) ;
      err << "Node[" << nodeOp.getNodeName() << "] Operation[" << op << "]\n"
	  << "\tvalue:" ;
      debugPrintObj<Value>(v1,err) ;
      err << "\tcurrent clk:" ;
      clockRepository.getClock(v1).debugPrint(err) ;
      err << "\ttarget clk:" ;
      clk1.debugPrint(err) ;
      err << "\n" ;
      err << "\tvalue:" ;
      debugPrintObj<Value>(v2,err) ;
      err << "\tcurrent clk:" ;
      clockRepository.getClock(v2).debugPrint(err) ;
      err << "\ttarget clk=" ;
      clk2.debugPrint(err) ;
      err << "\n" ;
      err << "Current clock solving context:\n" ;
      clockRepository.printClocksOnValues(err,2) ;
    }


    //============================================================
    // Here start the unification routines
    
    LogicalResult ClockAnalysis::whenAnalyse(Operation*op) {
      assert(isa<WhenOp>(op)) ;
      WhenOp whenOp(op) ;
      // Data that is perennial
      Cond<Type> condType = whenOp.getCondType() ;
      assert(condType.isNormal()) ;
      Value dataInVal = whenOp.getDataInput() ;      
      if(condType.getType() == CondDataType) {
	// Enforce clock equality between the condition and data inputs
	// since they are both present
	Value condVal = whenOp.getCondValue() ;
	Clock& condClock = clockRepository.getClock(condVal);
	Clock& dataInClock = clockRepository.getClock(dataInVal);
	if(!enforceClockEquality(condClock,dataInClock)){
	  Operation* op = whenOp.getOperation() ;
	  unificationError(*op,condVal,dataInVal,condClock,dataInClock) ;
	  return failure() ;
	}
      }
      // Enforce the subclock relation between the inputs and
      // the output
      {
	// Re-read this one, as it may have changed due to unification
	Clock& dataInClock = clockRepository.getClock(dataInVal);
	Value resultValue = whenOp.getResult() ;
	Clock& resultClock = clockRepository.getClock(resultValue) ;
	if(condType.getType() == CondDataType) {
	  Value condVal = whenOp.getCondValue() ;
	  bool whenotFlag = condType.getWhenotFlag() ;
	  Cond<Value> cond(condVal,whenotFlag) ;
	  WhenClock& subClock = clockRepository.buildWhenClock(dataInClock,cond) ;
	  if(!enforceClockEquality(subClock,resultClock)) {
	    Operation* op = whenOp.getOperation() ;
	    unificationError(*op,condVal,resultValue,subClock,resultClock) ;
	    return failure() ;
	  }
	} else {
	  KPeriodic word = condType.getWord() ;
	  Cond<Value> cond(word) ;
	  WhenClock& subClock = clockRepository.buildWhenClock(dataInClock,cond) ;
	  if(!enforceClockEquality(subClock,resultClock)) {
	    Operation* op = whenOp.getOperation() ;
	    unificationError(*op,dataInVal,resultValue,subClock,resultClock) ;
	    return failure() ;
	  }
	}
      }
      return success() ;
    }
    
    LogicalResult ClockAnalysis::mergeAnalyse(Operation*op) {
      assert(isa<MergeOp>(op)) ;
      MergeOp mergeOp = dyn_cast<MergeOp>(op) ;
      llvm::raw_fd_ostream err(2,false,true) ;
      if(verboseLevel>2) err << "  >>>>>>mergeAnalyse(begin)\n" ;
      // Data that is perennial
      Cond<Type> condType = mergeOp.getCondType() ;
      assert(condType.isNormal());
      Value trueInVal = mergeOp.getTrueInput() ;      
      Value falseInVal = mergeOp.getFalseInput() ;
      Value resultVal = mergeOp.getResult() ;
      if(condType.getType() == CondDataType) {
	// Enforce clock equality between the condition and output
	// variables, since they are both present
	Value condVal = mergeOp.getCondValue() ;
	Clock& condClock = clockRepository.getClock(condVal);
	Clock& resultClock = clockRepository.getClock(resultVal);
	if(!enforceClockEquality(condClock,resultClock)){
	  Operation* op = mergeOp.getOperation() ;
	  unificationError(*op,condVal,resultVal,condClock,resultClock) ;
	  return failure() ;
	}
      }
      if(verboseLevel>2) err << "  >>>>>>mergeAnalyse(1)\n" ;
      // Enforce the subclock relation between the true input and
      // the output
      {
	// Re-read this one, as it may have changed due to unification
	Clock& trueInClock = clockRepository.getClock(trueInVal);
	Clock& resultClock = clockRepository.getClock(resultVal) ;
	if(condType.getType() == CondDataType) {
	  Value condVal = mergeOp.getCondValue() ;
	  Cond<Value> cond(condVal,false) ;
	  WhenClock& subClock = clockRepository.buildWhenClock(resultClock,cond) ;
	  if(!enforceClockEquality(subClock,trueInClock)){
	    Operation* op = mergeOp.getOperation() ;
	    unificationError(*op,condVal,trueInVal,subClock,trueInClock) ;
	    return failure() ;
	  }
	} else {
	  KPeriodic word = condType.getWord() ;
	  Cond<Value> cond(word) ;
	  WhenClock& subClock = clockRepository.buildWhenClock(resultClock,cond) ;
	  if(!enforceClockEquality(subClock,trueInClock)){
	    Operation* op = mergeOp.getOperation() ;
	    unificationError(*op,resultVal,trueInVal,subClock,trueInClock) ;
	    return failure() ;
	  }
	}
      }
      if(verboseLevel>2) err << "  >>>>>>mergeAnalyse(2)\n" ;
      // Enforce the subclock relation between the false input and
      // the output
      {
	// Re-read this one, as it may have changed due to unification
	Clock& falseInClock = clockRepository.getClock(falseInVal);
	Clock& resultClock = clockRepository.getClock(resultVal) ;
	if(condType.getType() == CondDataType) {
	  if(verboseLevel>2) err << "  >>>>>>mergeAnalyse(2.1)\n" ;
	  Value condVal = mergeOp.getCondValue() ;
	  Cond<Value> cond(condVal,true) ; // XXX
	  WhenClock& subClock = clockRepository.buildWhenClock(resultClock,cond) ;
	  if(verboseLevel>2) err << ">>>>>>mergeAnalyse(2.1.1)\n" ;
	  if(!enforceClockEquality(subClock,falseInClock)) {
	    Operation* op = mergeOp.getOperation() ;
	    unificationError(*op,condVal,falseInVal,subClock,falseInClock) ;
	    return failure() ;
	  }
	  if(verboseLevel>2) err << "  >>>>>>mergeAnalyse(2.1.2)\n" ;
	} else {
	  if(verboseLevel>2) err << "  >>>>>>mergeAnalyse(2.2)\n" ;
	  KPeriodic word = condType.getWord().buildComplement() ;
	  Cond<Value> cond(word) ;
	  WhenClock& subClock = clockRepository.buildWhenClock(resultClock,cond) ;
	  if(verboseLevel>2) err << "  >>>>>>mergeAnalyse(2.2.1)\n" ;
	  if(!enforceClockEquality(subClock,falseInClock)){
	    if(verboseLevel>2) err << "  >>>>>>mergeAnalyse(2.2.3)\n" ;
	    Operation* op = mergeOp.getOperation() ;
	    unificationError(*op,resultVal,falseInVal,subClock,falseInClock) ;
	    return failure() ;
	  }
	  if(verboseLevel>2) err << "  >>>>>>mergeAnalyse(2.2.2)\n" ;
	}
      }
      if(verboseLevel>2) err << "  >>>>>>mergeAnalyse(end)\n" ;
      return success() ;
    }

    // This routine will unify all the clocks of all inputs and outputs.
    // It does so by putting all input and output values in a list, and
    // then iteratively unify the first element of the list with all of
    // the following.
    // NOTE: This is the analysis applied to "x = y fby z", meaning that the
    // clock of y will be made equal to that of x and z
    // TODO: make sure it works with node instances, even assuming that
    // the homogeneity requirement is correct.
    LogicalResult ClockAnalysis::simpleAnalyse(Operation* op) {
      // Start by putting all operand and result values in a
      // single list.
      list<Value> valueList ;
      {
	for(Value v : op->getOperands()) // Of type OperandRange
	  valueList.push_back(v) ; 
	for(OpResult o : op->getResults()) // Of type ResultRange
	  valueList.push_back(o) ;
      }
      return synchronizeList(valueList);
    }

    // Pre makes the output present at all cycles of the input
    // save the first
    LogicalResult ClockAnalysis::preAnalyse(Operation*op) {
      assert(isa<PreOp>(op)) ;
      PreOp preOp(op) ;
      llvm::raw_fd_ostream err(2,false,true) ;
      if(verboseLevel>2) err << "  >>>>>>preAnalyse(begin)\n" ;

      // 
      Cond<Value> preCond(KPeriodic::preWord) ;
      Value operandVal = preOp.getOperand() ;
      Value resultVal = preOp.getResult() ;
      Clock& operandClock = clockRepository.getClock(operandVal);
      Clock& resultClock  = clockRepository.getClock(resultVal);
      WhenClock& subClock =
	clockRepository.buildWhenClock(operandClock,preCond) ;
      if(!enforceClockEquality(subClock,resultClock)) {
	Operation* op = preOp.getOperation() ;
	unificationError(*op,operandVal,resultVal,operandClock,resultClock) ;
	return failure() ;
      }
      return success() ;
    }
    
    // Yield makes each input of the enclosing region the pre of the
    // corresponding output.
    LogicalResult ClockAnalysis::yieldAnalyse(Operation*op) {
      // assert(isa<YieldOp>(op)) ;
      // YieldOp yieldOp(op);
      
      // llvm::raw_fd_ostream err(2,false,true) ;
      // if(verboseLevel>2) err << "  >>>>>>yieldAnalyse(begin)\n" ;

      // // 
      // Cond<Value> preCond(KPeriodic::preWord) ;
      // for(auto e: llvm::zip(yieldOp.getStates(),nodeOp.getStates())) {
      // 	Value yieldVal = std::get<0>(e) ;
      // 	Value nodeVal = std::get<1>(e) ;
      // 	Clock& yieldClock = clockRepository.getClock(yieldVal);
      // 	Clock& nodeClock = clockRepository.getClock(nodeVal);
      // 	WhenClock& subClock = clockRepository.buildWhenClock(yieldClock,preCond) ;
      // 	if(!enforceClockEquality(subClock,nodeClock)) {
      // 	  Operation* op = yieldOp.getOperation() ;
      // 	  unificationError(*op,yieldVal,nodeVal,yieldClock,nodeClock) ;
      // 	  return failure() ;
      // 	}
      // }
      return success() ;
    }

    LogicalResult ClockAnalysis::analyseOp(Operation* op) {
      for (Region& r : op->getRegions()) {
	for (Operation& insideOp : *(r.begin())) {
	  return analyseOp(&insideOp);
	}
      }
      llvm::raw_fd_ostream err(2,false,true) ;
      if(verboseLevel>2) err << ">>>>>ClockAnalysis::analyseOp called on operation "
	  << (*op) << "<<<\n" ;

      if (isa<WhenOp>(op)) {
	if(verboseLevel>2) err << "  >>>>>>WhenOp\n" ;
	WhenOp whenOp(op) ;
	return whenAnalyse(whenOp);
      } else if(isa<MergeOp>(op)) {
	if(verboseLevel>2) err << "  >>>>>>MergeOp\n" ;
	MergeOp mergeOp = dyn_cast<MergeOp>(op) ;
	return mergeAnalyse(mergeOp);
      } else if(isa<YieldOp>(op)) {
	if(verboseLevel>2) err << "  >>>>>>YieldOp\n" ;
	YieldOp yieldOp(op) ; // = cast<YieldOp>(op);
	return yieldAnalyse(yieldOp);
      } else if(isa<PreOp>(op)) {
	if(verboseLevel>2) err << "  >>>>>>PreOp\n" ;
	PreOp preOp(op) ;
	return preAnalyse(preOp);
      } else if (isa<ConstantOp>(op)
		 || isa<sync::TickOp>(op) || isa<sync::SyncOp>(op)) {
	if(verboseLevel>2) err << "  >>>>>>ConstantOp\n" ;
	// For a ConstantOp, there is nothing to do.
	// Unification will be performed by the users of the
	// constant.
	// TODO: currently, I do not allow the use of:
	// - constants declared outside the region
	// - constants that are used on two different clocks
	// I should explicitly allow them through analysis, by
	// annotating them and excluding them from analysis.
	return success() ;
      } else {
	if(verboseLevel>2) err << ">>>>>>>>Simple Op\n" ;
	// All other operations (function calls, operation
	// instances, FbyOp) are subject to the simple analysis,
	// which unifies all clocks of inputs and outputs (and
	// state).
	// TODO: when I add hierarchy, I have to make special cases
	// for pssa.condact and pssa.yield, which must be included in
	// the hierarchy. BTW, hierarchical traversal should only be
	// present in pssa.condact.
        return simpleAnalyse(op);
      }
    }

    void ClockAnalysis::initializeClocks(Operation* op) {
      for (Region& r : op->getRegions()) {
	for (Operation& innerOp : *(r.begin())) {
	  initializeClocks(&innerOp);
	}
      }
      for (Value res : op->getOpResults()) {
	clockRepository.initializeClock(res);
      }
    }
    
    LogicalResult ClockAnalysis::analyse() {
      assert(!analyseDone);
      
      // Step 1: Assign a free clock to all SSA values.
      // - Note 1: This must be done before analysis, because
      //   operations may not be ordered by dominance.
      // - Note 2: Since the region cannot use external values,
      //   I only have to cover inputs, states, and the outputs
      //   of operations.
      // TODO: When I add hierarchy, variables must be assigned
      // hierarchically (do I have to disambiguate?).
      for (Value arg   : nodeOp.getInputs()) {
	clockRepository.initializeClock(arg);
      }
      for (Value arg   : nodeOp.getStatics()) {
	clockRepository.initializeClock(arg);
      }
      for (Value state : nodeOp.getStates()) {
	clockRepository.initializeClock(state);
      }
      for (Operation& op : *(nodeOp.getBody().begin())) {
	initializeClocks(&op);
      }

      // Tag Values which wouldn't be clocked
      for (Value arg   : nodeOp.getStatics()) {
	Clock &c = clockRepository.getClock(arg);
	neverUnify.insert(&c);
      }

      // Create an unbuffered error printing utility
      llvm::raw_fd_ostream err(2,false,true) ;
      if(verboseLevel>2) {
	err << "ClockAnalysis::analyse called on node "
	    << nodeOp.getOperation()->template getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName()).getValue()
	    << "\n" 
	    << "\tInitial clock assignments:\n" ;
	clockRepository.printClocksOnValues(err,2) ;
      }

      // Step 2: Traverse the set of operations and perform unification
      // as required by the operation
      // TODO: Better error management.
      // - Do not crash upon analysis failure in one operation. Instead,
      //   print the error in each equation that does not pass the analysis.
      // - If the verbose flag is set, also print the evolution of the
      //   clocks during clock inference (for all operations, and for the
      //   final link of the base clock).
      for (Operation& op : *(nodeOp.getBody().begin())) {
	if(!succeeded(analyseOp(&op)))
	  return op.emitOpError() << "Error on operation" ;
	// Tracing execution
	if(verboseLevel>2) {
	  err << "\tAnalyzing<<" << op <<">> succeeded: Updated clocks:\n" ;	
	  clockRepository.printClocksOnValues(err,2) ;
	}
      }

      // Step 3: Unify remaining free clocks with the base clock
      clockRepository.substituteBaseClock() ;
      if(verboseLevel>2) {
	err << "\tClock analysis success. Final updated clocks:\n" ;	
	clockRepository.printClocksOnValues(err,2) ;
      }

      {
	unsigned i = 0;
	for (Value arg   : nodeOp.getInputs()) {
	  Clock &c = clockRepository.getClock(arg);
	  if (!isa<BaseClock>(&c)) {
	    return nodeOp.emitError() << "Input value #" << i
				      << " should be on base clock.";
	  }
	  i++;
	}
      }

      {
	unsigned i = 0;
	YieldOp yieldOp = nodeOp.getYield();
	for (Value output : yieldOp.getOutputs()) {
	  Clock &c = clockRepository.getClock(output);
	  if (isa<FreeClock>(&c)) {
	    return yieldOp.emitError() << "Output value #" << i
				       << " should not be free at the end of clock analysis.";
	  }
	  i++;
	}
      }

      analyseDone = true;
      return success() ;
    }

    ClockTree& ClockAnalysis::getClockTree() {
      assert(analyseDone);
      if(!treeBuilt) {
	for (Operation& op : *(nodeOp.getBody().begin())) {
	  buildClockTree(&op);
	}
	treeBuilt = true;
      }
      return tree;
    }

    void ClockAnalysis::buildClockTree(Operation* op) {
      assert(analyseDone);
      assert(!treeBuilt);
      for (Region& r : op->getRegions()) {
      	for (Operation& innerOp : *(r.begin())) {
      	  buildClockTree(&innerOp);
      	}
      }
      if (op->getNumResults() > 0) {
      	Clock& c = clockRepository.getClock(op->getResult(0));
      	tree.add(op, c);
      }
      else if (op->getNumOperands() > 0) {
      	Clock& c = clockRepository.getClock(op->getOperand(0));
      	tree.add(op, c);
      }
      else {
      	tree.add(op);
      }
    }

    LogicalResult ClockAnalysis::synchronizeList(list<Value> valueList) {
      // If the list is empty, there is nothing to do.
      if(valueList.size() > 0) {
	// Enforce clock equality, between the first list element
	// and all remaining ones, successively
	Value fst = valueList.front() ;
	valueList.pop_front() ; // Remove the first element from the list
	for(auto v : valueList){
	  if(!enforceClockEquality(clockRepository.getClock(fst),
				   clockRepository.getClock(v))) {
	    return failure() ;
	  }
	}
      }
      return success();
    }
  }
}
