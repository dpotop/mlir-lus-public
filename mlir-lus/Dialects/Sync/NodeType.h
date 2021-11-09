#ifndef SYNC_NODE_TYPES_H
#define SYNC_NODE_TYPES_H

#include "mlir/IR/Types.h"

namespace mlir {
  namespace sync {

    /// Node Type Storage and Uniquing (mimicked from TypeDetail.h)
    class NodeTypeStorage : public TypeStorage {
    private:
      // The number of inputs is stored in the inherited TypeStorage
      // field. Here, I need to store the number of states, the number
      // of results, and the array of types.
      unsigned numStatic ;
      unsigned numInputs ;
      unsigned numOutputs ;
      Type const *staticAndIO;
      
    public:
      // Accessors - attention, getSubclassData seems to provide access to
      // the number of inputs, which is stored in the inherited TypeStorage
      // field.
      unsigned getStaticNumber() const  { return numStatic ; }
      unsigned getInputsNumber() const  { return numInputs ; }
      unsigned getOutputsNumber() const  { return numOutputs ; }
      
      ArrayRef<Type> getStatic() const {
	return ArrayRef<Type>(staticAndIO, numStatic);
      }
      ArrayRef<Type> getInputs() const {
	return ArrayRef<Type>(staticAndIO + numStatic , numInputs);
      }
      ArrayRef<Type> getOutputs() const {
	return ArrayRef<Type>(staticAndIO + numStatic + numInputs,
			      numOutputs);
      }
      
    public:
      using KeyTy = std::tuple<ArrayRef<Type>,ArrayRef<Type>,ArrayRef<Type>> ;
      bool operator==(const KeyTy &key) const {
	return key == KeyTy(getStatic(), getInputs(), getOutputs());
      }
      
      // Construction does not duplicate the vector of types (potential
      // memory problems)
      NodeTypeStorage(unsigned numStatic,
		      unsigned numInputs,
		      unsigned numOutputs,
		      Type const *staticAndIO)
	: TypeStorage(),
	  numStatic(numStatic),
	  numInputs(numInputs),
	  numOutputs(numOutputs),
	  staticAndIO(staticAndIO) {}


      /// Construction.
      static NodeTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
	ArrayRef<Type> statics = std::get<0>(key) ;
	ArrayRef<Type> inputs = std::get<1>(key) ;
	ArrayRef<Type> outputs = std::get<2>(key) ;

	// Copy the inputs and results into the bump pointer.
	SmallVector<Type, 16> types;
	types.reserve(statics.size() + inputs.size() + outputs.size());
	types.append(statics.begin(),  statics.end());
	types.append(inputs.begin(), inputs.end());
	types.append(outputs.begin(), outputs.end());
	auto typesList = allocator.copyInto(ArrayRef<Type>(types));

	// Initialize the memory using placement new.
	return new (allocator.allocate<NodeTypeStorage>())
	  NodeTypeStorage(statics.size(),
			  inputs.size(),
			  outputs.size(),
			  typesList.data());
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
			  ArrayRef<Type> outputs) {
	return Base::get(context, statics, inputs, outputs);
      }

      // Input types.
      unsigned getNumStatic() const { return getImpl()->getStaticNumber(); }
      ArrayRef<Type> getStatics() const { return getImpl()->getStatic(); }
      Type getStatic(unsigned i) const { return getStatics()[i]; }

      // Input types.
      unsigned getNumInputs() const { return getImpl()->getInputsNumber(); }
      ArrayRef<Type> getInputs() const { return getImpl()->getInputs(); }
      Type getInputs(unsigned i) const { return getInputs()[i]; }

      // Output types.
      unsigned getNumOutputs() const { return getImpl()->getOutputsNumber(); }
      ArrayRef<Type> getOutputs() const { return getImpl()->getOutputs(); }
      Type getOutputs(unsigned i) const { return getOutputs()[i]; }

    };
  }
}

#endif


// Local Variables:
// mode: c++
// End:
