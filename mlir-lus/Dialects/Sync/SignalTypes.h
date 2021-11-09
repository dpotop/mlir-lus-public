#ifndef SYNC_SIGNAL_TYPES_H
#define SYNC_SIGNAL_TYPES_H

#include "mlir/IR/Types.h"
// Needed to include the RegionKindInterface and its trait
#include "mlir/IR/RegionKindInterface.h" 

namespace mlir {
  namespace sync {

    /// Node Type Storage and Uniquing (mimicked from TypeDetail.h)
    class SiginTypeStorage : public TypeStorage {
    private:
      const Type baseType ;
      
    public:
      // Accessors 
      const Type getType() const { return baseType ; }
      
    public:
      /// The hash key used for uniquing.
      using KeyTy = Type;
      bool operator==(const KeyTy &key) const {
	return key == getType() ;
      }
      
      // Construction does not duplicate the vector of types (potential
      // memory problems)
      SiginTypeStorage(const Type baseType)
	: TypeStorage(),
	  baseType(baseType) {}


      /// Construction.
      static SiginTypeStorage *construct(TypeStorageAllocator &allocator,
					  const KeyTy &key) {
	const Type baseType = key ;

	// Initialize the memory using placement new.
	return new (allocator.allocate<SiginTypeStorage>())
	  SiginTypeStorage(baseType);
      }      
    };

    
    
    /// Function types map from a list of inputs and a list of
    /// state variables to a list of results.
    class SiginType : public Type::TypeBase<SiginType,
					    Type,
					    SiginTypeStorage> {
    public:
      using Base::Base;
      
      // Build a NodeType object based on the inputs, state, and
      // results
      static SiginType get(MLIRContext *context,
			   const Type baseType) {
	return Base::get(context,
			 baseType) ;
      }
      
      // Input types.
      const Type getType() const { return getImpl()->getType(); }
    };

    class SigoutTypeStorage : public TypeStorage {
    private:
      const Type baseType ;
      
    public:
      // Accessors 
      const Type getType() const { return baseType ; }
      
    public:
      /// The hash key used for uniquing.
      using KeyTy = Type;
      bool operator==(const KeyTy &key) const {
	return key == getType() ;
      }
      
      // Construction does not duplicate the vector of types (potential
      // memory problems)
      SigoutTypeStorage(const Type baseType)
	: TypeStorage(),
	  baseType(baseType) {}


      /// Construction.
      static SigoutTypeStorage *construct(TypeStorageAllocator &allocator,
					 const KeyTy &key) {
	const Type baseType = key ;

	// Initialize the memory using placement new.
	return new (allocator.allocate<SigoutTypeStorage>())
	  SigoutTypeStorage(baseType);
      }      
    };
    
    class SigoutType : public Type::TypeBase<SigoutType,
					     Type,
					     SigoutTypeStorage> {
    public:
      using Base::Base;
      
      // Build a NodeType object based on the inputs, state, and
      // results
      static SigoutType get(MLIRContext *context,
			    const Type baseType) {
	return Base::get(context,
			 baseType) ;
      }
      
      // Input types.
      const Type getType() const { return getImpl()->getType(); }
    };
  }
}

#endif


// Local Variables:
// mode: c++
// End:
