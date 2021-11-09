#ifndef YIELD_TYPES_H
#define YIELD_TYPES_H

#include "mlir/IR/Types.h"

namespace mlir {
  namespace lus {

    /// Node Type Storage and Uniquing (mimicked from TypeDetail.h)
    class YieldTypeStorage : public TypeStorage {
    private:
      // The number of states is stored in the inherited TypeStorage
      // field. Here, I need to store the number of results and the
      // array of types for both states and results.
      unsigned numStates ;
      unsigned numOutputs ;
      Type const *statesAndOutputs;
      
    public:
      // Accessors - attention, getSubclassData seems to provide access to
      // the number of inputs, which is stored in the inherited TypeStorage
      // field.
      unsigned getStateNumber() const  { return numStates ; }
      unsigned getOutputNumber() const { return numOutputs ; }
      ArrayRef<Type> getStates() const {
	return ArrayRef<Type>(statesAndOutputs, numStates) ;
      }
      ArrayRef<Type> getOutputs() const {
	return ArrayRef<Type>(statesAndOutputs + numStates, numOutputs);
      }
      
    public:
      /// The hash key used for uniquing. 
      using KeyTy = std::pair<ArrayRef<Type>,ArrayRef<Type>> ;
      bool operator==(const KeyTy &key) const {
	return key == KeyTy(getStates(),getOutputs());
      }
      
      // Construction does not duplicate the vector of types (potential
      // memory problems)
      YieldTypeStorage(unsigned numStates,
		       unsigned numOutputs,
		       Type const *statesAndOutputs)
	: TypeStorage(),
	  numStates(numStates),
	  numOutputs(numOutputs),
	  statesAndOutputs(statesAndOutputs) {}


      /// Construction.
      static YieldTypeStorage *construct(TypeStorageAllocator &allocator,
					 const KeyTy &key) {
	ArrayRef<Type> states = std::get<0>(key) ;
	ArrayRef<Type> results= std::get<1>(key) ;

	// Copy the inputs and results into the bump pointer.
	SmallVector<Type, 16> types;
	types.reserve(states.size() + results.size());
	types.append(states.begin(),  states.end());
	types.append(results.begin(), results.end());
	auto typesList = allocator.copyInto(ArrayRef<Type>(types));

	// Initialize the memory using placement new.
	return new (allocator.allocate<YieldTypeStorage>())
	  YieldTypeStorage(states.size(), results.size(),typesList.data());
      }
      
    };
    
    
    
    ///=========================================================
    /// YieldType
    ///---------------------------------------------------------
    
    /// Function types map from a list of inputs and a list of
    /// state variables to a list of results.
    class YieldType : public Type::TypeBase<YieldType,
					    Type,
					    YieldTypeStorage> {
    public:
      using Base::Base;
      
      // Build a YieldType object based on the inputs, state, and
      // results
      static YieldType get(MLIRContext *context,
			   ArrayRef<Type> states,
			   ArrayRef<Type> results) {
	return Base::get(context,states,results);
      }
      
      // State types.
      unsigned getNumStates() const { return getImpl()->getStateNumber(); }
      ArrayRef<Type> getStates() const { return getImpl()->getStates(); }
      Type getState(unsigned i) const { return getStates()[i]; }

      // Output types.
      unsigned getNumOutputs() const { return getImpl()->getOutputNumber(); }
      ArrayRef<Type> getOutputs() const { return getImpl()->getOutputs(); } 
      Type getOutput(unsigned i) const { return getOutputs()[i]; }

    };
  }
}

#endif


// Local Variables:
// mode: c++
// End:
