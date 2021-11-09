#ifndef WHEN_TYPES_H
#define WHEN_TYPES_H

#include <tuple>
#include "mlir/IR/Types.h"
#include "TestCondition.h" // For the k-period words

namespace mlir {
  namespace lus {

    /// When Type Storage and Uniquing (mimicked from TypeDetail.h)
    class WhenTypeStorage : public TypeStorage {
    private:
      // Type of input and output data
      Type dataType ;
      // The test condition, which includes the whenot flag
      Cond<Type> condType ;
            
    public:
      // Accessors
      const Type getDataType() const  { return dataType ; }
      const Cond<Type> getCondType() const { return condType ; }
      
    public:
      /// The hash key used for uniquing.
      /// I did a dirty job here, by substituting an unsigned for a Boolean.
      /// I'm not even sure it works otherwise without creating some weird pointers.
      using KeyTy = std::pair<Type,Cond<Type>> ;
      bool operator==(const KeyTy &key) const {
	return key == KeyTy(dataType,condType) ;
      }
      
      // Construction does not duplicate the vector of types (potential
      // memory problems)
      WhenTypeStorage(const Type dataType,
		      const Cond<Type> condType) 
	: TypeStorage(),
	  dataType(dataType),
	  condType(condType)
      {}

      /// Construction.
      
      static WhenTypeStorage *construct(TypeStorageAllocator &allocator,
					const KeyTy &key) {
	
	const Type& dataType = std::get<0>(key) ; 
	const Cond<Type> condType = std::get<1>(key) ; 
	
	// Initialize the memory using placement new.
	return new (allocator.allocate<WhenTypeStorage>())
	  WhenTypeStorage(dataType,condType);
      }
      /*
      void debugPrint(raw_ostream&os) const {
	os << "BEGIN: WhenTypeStorage::debugPrint\n" ;
	os << dataType << "\n" ;
	condType.print(os); os << "\n" ;
	os << "END: WhenTypeStorage::debugPrint\n" ; 
      }
      */

      
    };

    ///=========================================================
    /// WhenType - used for the type storage of both WhenOp
    /// and MergeOp. This is possible, because I don't
    /// have to store the type of all I/O, but only the data type
    /// and the condition type.
    ///---------------------------------------------------------
    class WhenType : public Type::TypeBase<WhenType,
					   Type,
					   WhenTypeStorage> {
    public:
      using Base::Base;
      
      // Build a NodeType object based on the inputs, state, and
      // results
      static WhenType get(MLIRContext *context,
			  Type& dataType,
			  Cond<Type>& condType) {
	return Base::get(context,dataType,condType);
      }
      
      // Accessors
      const Type getDataType() const { return getImpl()->getDataType(); }
      const Cond<Type> getCondType() const { return getImpl()->getCondType(); }

      /*
      void debugPrint(raw_ostream&os) const {
	os << "BEGIN: WhenType::debugPrint\n" ;
	getImpl() -> debugPrint(os) ;
	os << "END: WhenType::debugPrint\n" ; 
      }
      */
    };
  }
}

#endif


// Local Variables:
// mode: c++
// End:
