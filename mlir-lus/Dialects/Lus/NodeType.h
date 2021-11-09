#ifndef NODE_TYPES_H
#define NODE_TYPES_H

#include "mlir/IR/Types.h"
// Needed to include the RegionKindInterface and its trait
#include "mlir/IR/RegionKindInterface.h" 

namespace mlir {
  namespace lus {

    /// Node Type Storage and Uniquing (mimicked from TypeDetail.h)
    class NodeTypeStorage : public TypeStorage {
    private:
      // The number of inputs is stored in the inherited TypeStorage
      // field. Here, I need to store the number of states, the number
      // of results, and the array of types.
      unsigned numStatic ;
      unsigned numInputs ;
      unsigned numStates ;
      unsigned numResults;
      Type const *staticAndInputsAndStatesAndResults;
      RegionKind regionKind ;
      
    public:
      // Accessors - attention, getSubclassData seems to provide access to
      // the number of inputs, which is stored in the inherited TypeStorage
      // field.
      RegionKind getRegionKind() const { return regionKind ; }
      unsigned getStaticNumber() const  { return numStatic ; }
      unsigned getInputNumber() const  { return numInputs ; }
      unsigned getStateNumber() const  { return numStates ; }
      unsigned getResultNumber() const { return numResults ; }
      
      ArrayRef<Type> getStatic() const {
	return ArrayRef<Type>(staticAndInputsAndStatesAndResults, numStatic);
      }
      ArrayRef<Type> getInputs() const {
	return ArrayRef<Type>(staticAndInputsAndStatesAndResults + numStatic , numInputs);
      }
      ArrayRef<Type> getStates() const {
	return ArrayRef<Type>(staticAndInputsAndStatesAndResults + numStatic + numInputs , numStates);
      }
      ArrayRef<Type> getResults() const {
	return ArrayRef<Type>(staticAndInputsAndStatesAndResults + numStatic + numInputs + numStates, numResults);
      }
      
    public:
      /// The hash key used for uniquing.
      static inline unsigned regionKindToUnsigned(const RegionKind rk) {
	switch(rk) {
	case RegionKind::SSACFG: return 54 ;
	case RegionKind::Graph: return 75 ;
	default: assert(false) ;
	}
      }
      static inline RegionKind unsignedToRegionKind(unsigned rk) {
	switch(rk) {
	case 54: return RegionKind::SSACFG ;
	case 75: return RegionKind::Graph ;
	default: assert(false) ;
	}
      }
      using KeyTy = std::tuple<ArrayRef<Type>,ArrayRef<Type>,ArrayRef<Type>,ArrayRef<Type>,unsigned> ;
      bool operator==(const KeyTy &key) const {
	return key == KeyTy(getStatic(),getInputs(),getStates(),getResults(),
			    regionKindToUnsigned(getRegionKind()));
      }
      
      // Construction does not duplicate the vector of types (potential
      // memory problems)
      NodeTypeStorage(unsigned numStatic,
		      unsigned numInputs,
		      unsigned numStates,
		      unsigned numResults,
		      Type const *staticAndInputsAndStatesAndResults,
		      RegionKind regionKind)
	: TypeStorage(),
	  numStatic(numStatic),
	  numInputs(numInputs),
	  numStates(numStates),
	  numResults(numResults),
	  staticAndInputsAndStatesAndResults(staticAndInputsAndStatesAndResults),
      	  regionKind(regionKind) {}


      /// Construction.
      static NodeTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
	ArrayRef<Type> statics = std::get<0>(key) ;
	ArrayRef<Type> inputs = std::get<1>(key) ;
	ArrayRef<Type> states = std::get<2>(key) ;
	ArrayRef<Type> results= std::get<3>(key) ;
	RegionKind regionKind = unsignedToRegionKind(std::get<4>(key)) ;

	// Copy the inputs and results into the bump pointer.
	SmallVector<Type, 16> types;
	types.reserve(statics.size() + inputs.size() + states.size() + results.size());
	types.append(statics.begin(),  statics.end());
	types.append(inputs.begin() ,  inputs.end());
	types.append(states.begin() ,  states.end());
	types.append(results.begin(),  results.end());
	auto typesList = allocator.copyInto(ArrayRef<Type>(types));

	// Initialize the memory using placement new.
	return new (allocator.allocate<NodeTypeStorage>())
	  NodeTypeStorage(statics.size(),
			  inputs.size(),
			  states.size(),
			  results.size(),
			  typesList.data(),
			  regionKind);
      }
      
    };

    
    
    ///=========================================================
    /// NodeType
    ///---------------------------------------------------------
    
    /// Function types map from a list of inputs and a list of
    /// state variables to a list of results.
    class NodeType : public Type::TypeBase<NodeType,
					   Type,
					   NodeTypeStorage> {
    public:
      using Base::Base;
      
      // Build a NodeType object based on the inputs, state, and
      // results
      static NodeType get(MLIRContext *context,
			  ArrayRef<Type> statics,
			  ArrayRef<Type> inputs,
			  ArrayRef<Type> states,
			  ArrayRef<Type> results,
			  RegionKind regionKind) {
	return Base::get(context,
			 statics,inputs,states,results,
			 NodeTypeStorage::regionKindToUnsigned(regionKind));
      }

      // Input types.
      unsigned getNumStatic() const { return getImpl()->getStaticNumber(); }
      ArrayRef<Type> getStatics() const { return getImpl()->getStatic(); }
      Type getStatic(unsigned i) const { return getStatics()[i]; }

      // Input types.
      unsigned getNumInputs() const { return getImpl()->getInputNumber(); }
      ArrayRef<Type> getInputs() const { return getImpl()->getInputs(); }
      Type getInput(unsigned i) const { return getInputs()[i]; }

      // State types.
      unsigned getNumStates() const { return getImpl()->getStateNumber(); }
      ArrayRef<Type> getStates() const { return getImpl()->getStates(); }
      Type getState(unsigned i) const { return getStates()[i]; }

      // Result types.
      unsigned getNumResults() const { return getImpl()->getResultNumber(); }
      ArrayRef<Type> getResults() const { return getImpl()->getResults(); } 
      Type getResult(unsigned i) const { return getResults()[i]; }

      // Region Kind
      RegionKind getRegionKind() const { return getImpl()->getRegionKind() ; }
      
    };
  }
}

#endif


// Local Variables:
// mode: c++
// End:
