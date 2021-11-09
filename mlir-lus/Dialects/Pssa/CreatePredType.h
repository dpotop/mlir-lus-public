// -*- C++ -*- //

#ifndef CREATE_PRED_TYPE_H
#define CREATE_PRED_TYPE_H

#include "mlir/IR/DialectImplementation.h"
#include "../Lus/TestCondition.h"

using namespace mlir::lus;

namespace mlir {
  namespace pssa {
    
    class CreatePredTypeStorage : public TypeStorage {
    private:
      Cond<Type> condType;
    public:
      
      /// Accessors
      
      const bool isData() { return condType.getType() == CondDataType; }
      const bool isKPeriodic() { return condType.getType() == CondKPType; }
      const KPeriodic getWord() {
    	assert(isKPeriodic());
    	return condType.getWord();
      }

      /// Uniquing
      
      using KeyTy = Cond<Type> ;
      bool operator==(const KeyTy &key) const {
	return key == KeyTy(condType) ;
      }

      ///Construction
      
      CreatePredTypeStorage(const Cond<Type>& condType)
	: TypeStorage(), condType(condType) {}

      static CreatePredTypeStorage *construct(TypeStorageAllocator &allocator,
    					      const KeyTy &key) {
	return new (allocator.allocate<CreatePredTypeStorage>())
	  CreatePredTypeStorage(key);
      }
    };

    class CreatePredType: public Type::TypeBase< CreatePredType,
						 Type,
						 CreatePredTypeStorage > {
    public:
      using Base::Base;

      static CreatePredType get(MLIRContext *context,Cond<Type> condType) {
    	return Base::get(context, condType);
      }

      const bool isData() { return getImpl()->isData(); }
      const bool isKPeriodic() { return getImpl()->isKPeriodic(); }
      const KPeriodic getWord() { return getImpl()->getWord(); }
    };
  }
}

#endif
