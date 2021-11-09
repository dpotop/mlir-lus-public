// -*- C++ -*- //

#ifndef COND_TO_PRED_H
#define COND_TO_PRED_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <unordered_map>
#include "ValueHash.h"
#include "../../Dialects/Lus/KPeriodic.h"
#include "../../Dialects/Lus/TestCondition.h"

using namespace std;

namespace mlir {

  using namespace lus;
  
  struct CondToPred {
  private:
    unordered_map<Value, Operation*, ValueHash> val_to_pred;
    unordered_map<KPeriodic, Operation*, KPeriodicHash> kperiod_to_pred;
  public:
    unsigned count(Value v) { return val_to_pred.count(v); }
    unsigned count(KPeriodic kp) { return kperiod_to_pred.count(kp); }
    Operation* get(Value v) {
      assert(count(v) > 0);
      return val_to_pred[v];
    }
    void set(Value v, Operation* op) {
      val_to_pred[v] = op;
    }
    Operation* get(KPeriodic kp) {
      assert(count(kp) > 0);
      return kperiod_to_pred[kp];
    }
    void set(KPeriodic kp, Operation* op) { kperiod_to_pred[kp] = op; }

    void print_keys() {
      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stdout", err);
      stream << "* Registered condition values:\n";
      for (std::pair<Value, Operation*> p : val_to_pred) {
	stream << " -> " << p.first << "\n";
      }
      for (std::pair<KPeriodic, Operation*> p : kperiod_to_pred) {
	stream << " -> " ;
	p.first.print(stream);
	stream << "\n";
      }
    }

    Value getValue(Cond<Value> cond) {
      Value predicate;
      if (cond.getType() == CondDataType) {
	if (cond.getWhenotFlag()) {
	  predicate = get(cond.getData())->getResult(1);
	}
	else {
	  predicate = get(cond.getData())->getResult(0);
	}
      }
      else if (cond.getType() == CondKPType) {
	predicate = get(cond.getWord())->getResult(0);
      }
      else {
	assert(false);
      }
      return predicate;
    }
  };
}

#endif
