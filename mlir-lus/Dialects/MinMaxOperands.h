// -*- C++ -*- //

#ifndef MIN_MAX_OPERANDS
#define MIN_MAX_OPERANDS

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
  namespace OpTrait {

    namespace impl {
      
      inline LogicalResult verifyMinMaxOperands(Operation *op,
						unsigned numOperandsMin,
						unsigned numOperandsMax) {
	unsigned num_ops = op->getNumOperands() ;
	if ((num_ops < numOperandsMin)||(num_ops > numOperandsMax)) {
	  return op->emitOpError() << "has " << num_ops
				   << " operands but expected between "
				   << numOperandsMin << " and "
				   << numOperandsMax ;
	}
	return success();
      }
    }
    
    template <unsigned Min,unsigned Max> class MinMaxOperands {
      
    public:
      
      template <typename ConcreteType>
      class Impl :
	public detail::MultiOperandTraitBase <ConcreteType,
					      MinMaxOperands<Min,Max>::Impl> {
      public:
	
	static LogicalResult verifyTrait(Operation *op) {
	  return impl::verifyMinMaxOperands(op, Min, Max);
	}
	
      };
    };
    
  }
}

#endif
