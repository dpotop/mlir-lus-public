#include "mlir/Pass/Pass.h" // For ModuleOp
#include "RemoveSomeUndefs.h"
#include "../../Dialects/Sync/UndefOp.h"
#include <algorithm>

namespace mlir {
  namespace pssa {

    void RemoveSomeUndefs::operator() (ModuleOp mOp) {
      moduleOp = mOp;
      apply(mOp);
      for (Operation *op: dishes) {
      	op->erase();
      }
    }
    
    void RemoveSomeUndefs::apply(Operation *op) {
      if (isa<FuncOp>(op)
	  && dyn_cast<FuncOp>(op).isExternal())
	return;

      for (Region &r: op->getRegions()) {
	for (Block &b: r) {
	  for (Operation &deeperOp: b) {
	    apply(&deeperOp);
	  }
	}
      }
      
      if (isa<sync::UndefOp>(op)) {
	sync::UndefOp undefOp = dyn_cast<sync::UndefOp>(op);
	OpBuilder builder(undefOp);
	Location loc = undefOp.getLoc();
	Type t = undefOp.getResult().getType();
	Value v;
	if (t.isa<TensorType>()) {
	  FuncOp f;
	  if (undefFuncs.count(t) == 0) {
	    OpBuilder funcBuild(&moduleOp.getBodyRegion());
	    std::string funcName;
	    llvm::raw_string_ostream stream(funcName);
	    stream << "undef_tensor";
	    for (int64_t dim: t.cast<TensorType>().getShape()) {
	      stream << "_" << dim;
	    }
	    stream << "_" << t.cast<TensorType>().getElementType();
	    funcName.erase(std::remove(funcName.begin(),
				       funcName.end(), '<'),
			   funcName.end());
	    funcName.erase(std::remove(funcName.begin(),
				       funcName.end(), '>'),
			   funcName.end());
	    OperationState funcState(loc, FuncOp::getOperationName());
	    FunctionType ft = funcBuild.getFunctionType({}, t);
	    StringAttr visibility = builder.getStringAttr("private") ;
	    FuncOp::build(funcBuild,
			  funcState,
			  StringAttr::get(builder.getContext(), funcName),
			  TypeAttr::get(ft),
			  visibility);
	    Operation *funcOpPtr = funcBuild.createOperation(funcState);
	    f = dyn_cast<FuncOp>(funcOpPtr);
	    undefFuncs[t] = f;
	  }
	  else {
	    f = undefFuncs[t];
	  }
	  OperationState callState(loc, CallOp::getOperationName());
	  CallOp::build(builder, callState, f);
	  Operation *callOpPtr = builder.createOperation(callState);
	  undefOp.replaceAllUsesWith(callOpPtr);
	  dishes.push_back(undefOp);
	}
	else {
	  FuncOp f;
	  if (undefFuncs.count(t) == 0) {
	    OpBuilder funcBuild(&moduleOp.getBodyRegion());
	    std::string funcName;
	    llvm::raw_string_ostream stream(funcName);
	    stream << "undef_" << t;
	    // for (int64_t dim: t.cast<TensorType>().getShape()) {
	    //   stream << "_" << dim;
	    // }
	    // stream << "_" << t.cast<TensorType>().getElementType();
	    FunctionType ft = funcBuild.getFunctionType({}, t);
	    OperationState funcState(loc, FuncOp::getOperationName());
	    StringAttr visibility = builder.getStringAttr("private") ;
	    FuncOp::build(funcBuild,
			  funcState,
			  StringAttr::get(builder.getContext(), funcName),
			  TypeAttr::get(ft),
			  visibility);

	    Operation *funcOpPtr = funcBuild.createOperation(funcState);
	    f = dyn_cast<FuncOp>(funcOpPtr);
	    undefFuncs[t] = f;
	  }
	  else {
	    f = undefFuncs[t];
	  }
	  OperationState callState(loc, CallOp::getOperationName());
	  CallOp::build(builder, callState, f);
	  Operation *callOpPtr = builder.createOperation(callState);
	  undefOp.replaceAllUsesWith(callOpPtr);
	  dishes.push_back(undefOp);
	}
      }
      // else if (isa<PsiOp>(op)) {
      // 	PsiOp psiOp = dyn_cast<PsiOp>(op);
      // 	OpBuilder builder(psiOp);
      // 	Location loc = psiOp.getLoc();
      // 	Value prevBranch = psiOp.getBranches()[psiOp.getNumBranches() - 1];
      // 	for (int i = psiOp.getNumBranches() - 2 ; i >= 0 ; i--) {
      // 	  Value condRaw = psiOp.getConditions()[i];
      // 	  Value trueBranch = psiOp.getBranches()[i];
      // 	  Type t = trueBranch.getType();
      // 	  if (t.isa<TensorType>()) {
      // 	    FuncOp f;
      // 	    if (selectFuncs.count(t) == 0) {
      // 	      OpBuilder funcBuild(&moduleOp.getBodyRegion());
      // 	      Type indexType = funcBuild.getI1Type();
      // 	      std::string funcName;
      // 	      llvm::raw_string_ostream stream(funcName);
      // 	      stream << "select_tensor";
      // 	      for (int64_t dim: t.cast<TensorType>().getShape()) {
      // 		stream << "_" << dim;
      // 	      }
      // 	      stream << "_" << t.cast<TensorType>().getElementType();
      // 	      FunctionType ft = funcBuild.getFunctionType({indexType, t, t},
      // 							  t);
      // 	      OperationState funcState(loc, FuncOp::getOperationName());
      // 	      FuncOp::build(funcBuild, funcState, funcName, ft);
      // 	      Operation *funcOpPtr = funcBuild.createOperation(funcState);
      // 	      f = dyn_cast<FuncOp>(funcOpPtr);
      // 	      selectFuncs[t] = f;
      // 	    }
      // 	    else {
      // 	      f = selectFuncs[t];
      // 	    }
      // 	    OperationState callState(loc, CallOp::getOperationName());
      // 	    CallOp::build(builder, callState, f,
      // 			  {condRaw, trueBranch, prevBranch});
      // 	    Operation *callOpPtr = builder.createOperation(callState);
      // 	    prevBranch = callOpPtr->getResult(0);
      // 	  }
      // 	}
      // 	psiOp.getResult().replaceAllUsesWith(prevBranch);
      // 	dishes.push_back(psiOp);
      // }
    }
  }
}
