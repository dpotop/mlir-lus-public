#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/AsmState.h"
#include "TestCondition.h"

namespace mlir {
  namespace lus {

    // Helper function used with clocks and clock analysis.
    // Pretty-prints a value on a raw_ostream.
    template<> void debugPrintObj<Value>(Value&v,raw_ostream&err) {    
      // Determine the parent node of the value, and build a printing facility	
      Operation* op = v.getDefiningOp();
      if(op == NULL) {
	auto blk = v.getParentBlock() ;
	assert(blk != NULL) ;
	op = blk->getParentOp() ;
	assert(op != NULL) ;
      }
      while ((op != NULL) &&
	     (op->getParentOp() != NULL)) { op = op->getParentOp(); }
      AsmState state(op);
      v.printAsOperand(err,state) ;
    }
    template<> void debugPrintObj<Type>(Type&v,raw_ostream&err) {    
      // Determine the parent node of the value, and build a printing facility
      err << v ;
    }
    
    template<class TestObject>
    EmptyCondStorage<TestObject> Cond<TestObject>::emptyKey ;
    template<class TestObject>
    TombstoneCondStorage<TestObject> Cond<TestObject>::tombstoneKey ;

    template<class TestObject>
    const Cond<TestObject> Cond<TestObject>::getEmptyKey() {
      Cond<TestObject> res(&emptyKey) ;
      return res ;
    }
    template<class TestObject>
    const Cond<TestObject> Cond<TestObject>::getTombstoneKey() {
      Cond<TestObject> res(&tombstoneKey) ;
      return res ;
    }
    
    template class Cond<Value> ;
    template class Cond<Type> ;
    
  }
}
