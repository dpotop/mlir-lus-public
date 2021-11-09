#ifndef SIGNAL_TYPES_H
#define SIGNAL_TYPES_H

#include "mlir/IR/Types.h"

namespace mlir {
  namespace lus {

    class SignalTypeStorage : public TypeStorage {
    private:
      const Type baseType ;
      const bool isInputFlag ;
      
    public:
      const Type getType() const { return baseType ; }
      const bool isInput() const { return isInputFlag ; }
      
    public:
      using KeyTy = std::pair<Type,unsigned>;
      bool operator==(const KeyTy &key) const {
	return key == KeyTy(getType(),isInput()) ;
      }
      
      SignalTypeStorage(const Type baseType,
			bool isInputFlag)
	: TypeStorage(),
	  baseType(baseType),
	  isInputFlag(isInputFlag) {}


      static SignalTypeStorage *construct(TypeStorageAllocator &allocator,
					  const KeyTy &key) {
	const Type baseType = std::get<0>(key) ;
	const bool isInputFlag = std::get<1>(key) ;

	return new (allocator.allocate<SignalTypeStorage>())
	  SignalTypeStorage(baseType,isInputFlag);
      }      
    };
    
    class SignalType : public Type::TypeBase<SignalType,
					     Type,
					     SignalTypeStorage> {
    public:
      using Base::Base;
      
      static SignalType get(MLIRContext *context,
			    const Type baseType,
			    bool isInputFlag) {
	return Base::get(context,
			 baseType,
			 isInputFlag) ;
      }
      
      const Type getType() const { return getImpl()->getType(); }
      bool isInput() const { return getImpl()->isInput() ; }      
    };
  }
}

#endif


// Local Variables:
// mode: c++
// End:
