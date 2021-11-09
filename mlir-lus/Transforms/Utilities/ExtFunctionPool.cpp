#include "ExtFunctionPool.h"
#include "BufferGenerator.h"
#include "../../Dialects/Sync/SignalTypes.h"

namespace mlir {

  FuncOp ExtFunctionPool::schStart(Type st) {
    if (st.isa<sync::SiginType>()) {
      Type t = st.cast<sync::SiginType>().getType();
      Type bt = BufferGenerator::abstractBufferizedType(t);
      FunctionType ft = BufferGenerator::inputFunction(myBuilder, bt);
      std::string name = "sched_read_input_" + printable(bt);
      return build(name, ft);
    }
    else if (st.isa<sync::SigoutType>()) {
      Type t = st.cast<sync::SigoutType>().getType();
      Type bt = BufferGenerator::abstractBufferizedType(t);
      FunctionType ft = BufferGenerator::outputFunction(myBuilder, bt);
      std::string name = "sched_write_output_" + printable(bt);
      return build(name, ft);
    }
    else {
      assert(false);
    }
  }

  FuncOp ExtFunctionPool::schSetInstance() {
    std::string funcName = "sch_set_instance";
    Operation *funcOpPtr;
    if (funcOps.count(funcName) == 0) {
      OperationState funcState(myBuilder.getUnknownLoc(),
			       FuncOp::getOperationName());
      Type i32 = myBuilder.getI32Type();
      FunctionType startType = myBuilder.getFunctionType({i32}, {});
      FunctionType ft = myBuilder.getFunctionType({i32, startType, i32, i32},
						  {});
      StringAttr visibility = myBuilder.getStringAttr("private") ;
      FuncOp::build(myBuilder, funcState,
		    StringAttr::get(myBuilder.getContext(), funcName),
		    TypeAttr::get(ft),
		    visibility);
      funcOpPtr = myBuilder.createOperation(funcState);
      funcOps[funcName] = funcOpPtr;
    }
    else {
      funcOpPtr = funcOps[funcName];
    }
    return dyn_cast<FuncOp>(funcOpPtr);
  }

  FuncOp ExtFunctionPool::schSetIOInput(Type t) {
    Type i32 = myBuilder.getI32Type();
    FunctionType ft = myBuilder.getFunctionType({i32,i32,i32,i32,t},
						{});
    std::string name = "sched_set_input_" + printable(t);
    return build(name, ft);
  }

  FuncOp ExtFunctionPool::schSetIOOutput(Type t) {
    Type i32 = myBuilder.getI32Type();
    FunctionType ft = myBuilder.getFunctionType({i32,i32,i32,i32,t},
						{});
    std::string name = "sched_set_output_" + printable(t);
    return build(name, ft);
  }

  FuncOp ExtFunctionPool::inst() {
    std::string funcName = "inst";
    Operation *funcOpPtr;
    if (funcOps.count(funcName) == 0) {
      OperationState funcState(myBuilder.getUnknownLoc(),
			       FuncOp::getOperationName());
      FunctionType ft = myBuilder.getFunctionType({myBuilder.getI32Type()},
						  {});
      StringAttr visibility = myBuilder.getStringAttr("private") ;
      FuncOp::build(myBuilder, funcState,
		    StringAttr::get(myBuilder.getContext(), funcName),
		    TypeAttr::get(ft),
		    visibility);
      funcOpPtr = myBuilder.createOperation(funcState);
      funcOps[funcName] = funcOpPtr;
    }
    else {
      funcOpPtr = funcOps[funcName];
    }
    return dyn_cast<FuncOp>(funcOpPtr);
  }

  FuncOp ExtFunctionPool::build(std::string funcName, FunctionType ft) {
    Operation *funcOpPtr;
    if (funcOps.count(funcName) == 0) {
      OperationState funcState(myBuilder.getUnknownLoc(),
			       FuncOp::getOperationName());
      StringAttr visibility = myBuilder.getStringAttr("private") ;
      FuncOp::build(myBuilder, funcState,
		    StringAttr::get(myBuilder.getContext(), funcName),
		    TypeAttr::get(ft),
		    visibility);
      funcOpPtr = myBuilder.createOperation(funcState);
      funcOps[funcName] = funcOpPtr;
    }
    else {
      funcOpPtr = funcOps[funcName];
    }
    return dyn_cast<FuncOp>(funcOpPtr);
  }
  
  std::string ExtFunctionPool::name(std::string prefix, FunctionType ft) {
    std::string name = prefix;
    name += "_in";
    for (Type t: ft.getInputs()) {
      name += "_";
      name += printable(t);
    }
    name += "_out";
    for (Type t: ft.getResults()) {
      name += "_";
      name += printable(t);
    }
    return name;
  }
  
  std::string ExtFunctionPool::printable(Type t) {
    std::string name;
    llvm::raw_string_ostream stream(name);
    stream << t;
    name.erase(std::remove(name.begin(), name.end(), '<'), name.end());
    name.erase(std::remove(name.begin(), name.end(), '>'), name.end());
    name.erase(std::remove(name.begin(), name.end(), '?'), name.end());
    return name;
  }

}
