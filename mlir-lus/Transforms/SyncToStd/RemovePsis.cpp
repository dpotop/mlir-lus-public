#include "mlir/Pass/Pass.h" // For ModuleOp
#include "RemovePsis.h"
#include "../../Dialects/Sync/UndefOp.h"
#include "../../Dialects/Sync/SelectOp.h"
#include "../Utilities/OperationsAux.h"

namespace mlir {
  namespace pssa {

    void RemovePsis::operator() (ModuleOp mOp) {
      moduleOp = mOp;
      std::list<Operation*> topLevel;
      for (Operation &op : *(mOp.getRegion().begin())) {
	topLevel.push_back(&op);
      }
      for (Operation *op : topLevel) {
	if (isa<FuncOp>(op)
	    && dyn_cast<FuncOp>(op).isExternal()) {
	}
	else if (isa<FuncOp>(op)){
	  ConstantPool constantPool(dyn_cast<FuncOp>(op));
	  apply(op, constantPool);
	}
	else if (isa<sync::NodeOp>(op)){
	  ConstantPool constantPool(dyn_cast<sync::NodeOp>(op));
	  apply(op, constantPool);
	}
      }
      for (Operation *op: dishes) {
      	op->erase();
      }
    }
    
    void RemovePsis::apply(Operation *op, ConstantPool &constantPool) {

      for (Region &r: op->getRegions()) {
	for (Block &b: r) {
	  for (Operation &deeperOp: b) {
	    apply(&deeperOp, constantPool);
	  }
	}
      }
      
      if (isa<sync::SelectOp>(op)) {
	sync::SelectOp selectOp = dyn_cast<sync::SelectOp>(op);
	OpBuilder builder(selectOp);
	Location loc = selectOp.getLoc();

	Value cond = selectOp.getCondition();
	Value trueBranch = selectOp.getTrueBranch();
	Value falseBranch = selectOp.getFalseBranch();
	Type t = trueBranch.getType();
	Value res;

	if (t.isa<TensorType>()) {
	  FuncOp f;
	  if (selectFuncs.count(t) == 0) {
	    OpBuilder funcBuild(&moduleOp.getBodyRegion());
	    Type indexType = funcBuild.getI1Type();
	    std::string funcName;
	    llvm::raw_string_ostream stream(funcName);
	    stream << "select_tensor";
	    for (int64_t dim: t.cast<TensorType>().getShape()) {
	      stream << "_" << dim;
	    }
	    stream << "_" << t.cast<TensorType>().getElementType();
	    FunctionType ft = funcBuild.getFunctionType({indexType, t, t},
							t);
	    OperationState funcState(loc, FuncOp::getOperationName());
	    
	    StringAttr visibility = builder.getStringAttr("private") ;
	    FuncOp::build(funcBuild,
			  funcState,
			  StringAttr::get(builder.getContext(), funcName),
			  TypeAttr::get(ft),
			  visibility);

	    
	    Operation *funcOpPtr = funcBuild.createOperation(funcState);
	    f = dyn_cast<FuncOp>(funcOpPtr);
	    selectFuncs[t] = f;
	  }
	  else {
	    f = selectFuncs[t];
	  }
	  OperationState callState(loc, CallOp::getOperationName());
	  CallOp::build(builder, callState, f, {cond,trueBranch,falseBranch});
	  Operation *callOpPtr = builder.createOperation(callState);
	  res = callOpPtr->getResult(0);
	}
	else {
	  Value condition;
	  if (cond.getType().isInteger(1)) {
	    condition = cond;
	  }
	  else if (cond.getType().isInteger(32)) {
	    Value oneInt = constantPool.getOne(builder.getI32Type());
	    OperationState cmpState(loc, CmpIOp::getOperationName());
	    CmpIOp::build(builder, cmpState, builder.getI1Type(),
			  CmpIPredicate::eq, cond, oneInt);
	    Operation *cmpOpPtr = builder.createOperation(cmpState);
	    CmpIOp cmpOp = dyn_cast<CmpIOp>(cmpOpPtr);
	    condition = cmpOp.getResult();
	  }
	  else {
	    assert(false);
	  }
	  OperationState selectState(selectOp.getLoc(),
				     SelectOp::getOperationName());
	  SelectOp::build(builder, selectState, condition,
			  trueBranch, falseBranch);
	  Operation *selectOpPtr = builder.createOperation(selectState);
	  SelectOp selectOp = dyn_cast<SelectOp>(selectOpPtr);
	  res = selectOp.getResult();
	}
	selectOp.getResult().replaceAllUsesWith(res);
	dishes.push_back(selectOp);
      }
    }
  }
}
