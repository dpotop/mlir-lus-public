#include "NodeToFun.h"
#include "../Utilities/BufferGenerator.h"

namespace mlir {
  namespace sync {

    void NodeToFun::operator() (ModuleOp moduleOp) {
      
      extFunctionPool = new ExtFunctionPool(moduleOp);

      instantiateNodes(moduleOp);

      for (Operation &op : *(moduleOp.getBodyRegion().begin())) {
	if (isa<NodeOp>(&op) && instances.count(&op) == 0) {
	  NodeOp nodeOp = dyn_cast<NodeOp>(&op);
	  buildCoreFun(nodeOp);
	  // FuncOp funcOp = buildCoreFun(nodeOp);
	  // lowerIO(funcOp, funcOp);
	}
      }

      for (Operation *op : dishes) {
	op->erase();
      }

      delete extFunctionPool;
    }

    void NodeToFun::instantiateNodes(Operation *op) {
      for (Region &r: op->getRegions()) {
	if (!r.empty()) {
	  for (Operation &deeperOp: *r.begin()) {
	    instantiateNodes(&deeperOp);
	  }
	}
      }
      if (isa<InstOp>(op)) {
	InstOp instOp = dyn_cast<InstOp>(op);
	instantiate(instOp);
      }
    }

    void NodeToFun::instantiate(InstOp instOp) {

      OpBuilder builder(instOp);
      FuncOp coreFun = buildCoreFun(instOp);
      FuncOp startFun = buildStartFun(instOp);
      FuncOp instFun = buildInstFun(instOp, startFun);
      ConstantPool cpCF(coreFun);
      // lowerIO(coreFun, coreFun);
      ConstantPool cpInst(instOp.getOperation()->getParentRegion());
      Value instIdValue = cpInst.buildInt(builder.getI32Type(),
					  instOp.getId());
      SmallVector<Value, 4> prefixArgs;
      prefixArgs.push_back(instIdValue);
      instOpToCall(instOp, instFun);

      dishes.push_back(instOp);
    }
    
    FuncOp NodeToFun::buildStartFun(InstOp instOp) {

      OpBuilder builder(instOp);
      StringRef nodeName = instOp.getCalleeName();

      // Build the start function (no parameters)
      
      std::string startName = nodeName.str() + "_start";
      FunctionType ft = builder.getFunctionType({builder.getI32Type()},
						{});
      FuncOp funcOp = extFunctionPool->build(startName, ft);
      Block * b = funcOp.addEntryBlock();

      // Get signal types
      
      SmallVector<Type, 4> iosTypes;
      for (Type t : instOp.getArgOperands().getTypes()) {
	Type st = SiginType::get(builder.getContext(), t);
	iosTypes.push_back(st);
      }
      for (Type t : instOp.getResults().getTypes()) {
	Type st = SigoutType::get(builder.getContext(), t);
	iosTypes.push_back(st);
      }

      // Build function pointers
      
      ConstantPool constantPool(funcOp);
      SmallVector<Value, 4> callArgs;
      Value instIdValue = b->getArgument(0);
      callArgs.push_back(instIdValue);
      for (Type t : iosTypes) {
	FuncOp inFun = extFunctionPool->schStart(t);
	funcOp.getOperation()->moveAfter(inFun);
	Value inFunPtr = constantPool.buildFuncPointer(inFun);
	callArgs.push_back(inFunPtr);
      }

      // Call the main code and return

      builder.setInsertionPointToEnd(&funcOp.getBody().back());
      
      OperationState callState(instOp.getLoc(), CallOp::getOperationName());
      CallOp::build(builder, callState, nodeName, {}, callArgs);
      builder.createOperation(callState);
      
      OperationState returnState(instOp.getLoc(),
				 ReturnOp::getOperationName());
      ReturnOp::build(builder, returnState);
      builder.createOperation(returnState);

      return funcOp;
    }

    FuncOp NodeToFun::buildInstFun(InstOp instOp, FuncOp startFun) {
      StringRef nodeName = instOp.getCalleeName();

      OpBuilder builder(startFun);
      builder.setInsertionPointAfter(startFun);

      // Build the inst function : i64, input types, output types -> ()
      
      std::string instName = nodeName.str() + "_inst";
      SmallVector<Type, 4> paramTypes;
      paramTypes.push_back(builder.getI32Type());
      for (Type t : instOp.getArgOperands().getTypes()) {
	Type bt = BufferGenerator::abstractBufferizedType(t);
	paramTypes.push_back(bt);
      }
      for (Type t : instOp.getResults().getTypes()) {
	Type bt = BufferGenerator::abstractBufferizedType(t);
	paramTypes.push_back(bt);
      }
      FunctionType ft = builder.getFunctionType(paramTypes, {});
      FuncOp funcOp = extFunctionPool->build(instName, ft);
      Block *b = funcOp.addEntryBlock();
      builder.setInsertionPointToStart(b);

      // Set instance
      
      Value instIdValue = b->getArgument(0);
      ConstantPool constantPool(funcOp);
      Value startFunPtr = constantPool.buildFuncPointer(startFun);
      FuncOp setInstFunc = extFunctionPool->schSetInstance();
      const unsigned numInputs = instOp.getArgOperands().size();
      Value numInputsVal = constantPool.buildInt(builder.getI32Type(),
						 numInputs);
      const unsigned numOutputs = instOp.getResults().size();
      Value numOutputsVal = constantPool.buildInt(builder.getI32Type(),
						  numOutputs);
      OperationState callState(instOp.getLoc(), CallOp::getOperationName());
      CallOp::build(builder, callState, setInstFunc,
		    {instIdValue, startFunPtr, numInputsVal, numOutputsVal});
      builder.createOperation(callState);

      // Set inputs 
      
      for (unsigned i = 1; i < numInputs + 1; i++) {
	Value v = b->getArgument(i);
	Value position = constantPool.buildInt(builder.getI32Type(), i - 1);
	FuncOp setIOFunc = extFunctionPool->schSetIOInput(v.getType());
	Value numdims = constantPool.buildNumDims(v.getType());
	Value datasize = constantPool.buildDataSize(v.getType());
	OperationState callState(instOp.getLoc(), CallOp::getOperationName());
	CallOp::build(builder, callState, setIOFunc, {instIdValue, position,
						      numdims, datasize,
						      v});
	builder.createOperation(callState);
      }

      // Set outputs
      
      for (unsigned i = numInputs + 1; i < b->getNumArguments(); i++) {
	Value v = b->getArgument(i);
	Value position = constantPool.buildInt(builder.getI32Type(),
					       i - numInputs - 1);
	FuncOp setIOFunc = extFunctionPool->schSetIOOutput(v.getType());
	Value numdims = constantPool.buildNumDims(v.getType());
	Value datasize = constantPool.buildDataSize(v.getType());
	OperationState callState(instOp.getLoc(), CallOp::getOperationName());
	CallOp::build(builder, callState, setIOFunc, {instIdValue, position,
						      numdims, datasize,
						      v});
	builder.createOperation(callState);
      }

      // Call inst i.e. give control to scheduler
      
      FuncOp instFunc = extFunctionPool->inst();
      funcOp.getOperation()->moveAfter(instFunc);
      OperationState callInstState(instOp.getLoc(),
				   CallOp::getOperationName());
      CallOp::build(builder, callInstState, instFunc, {instIdValue});
      builder.createOperation(callInstState);

      OperationState returnState(instOp.getLoc(),
				 ReturnOp::getOperationName());
      ReturnOp::build(builder, returnState);
      builder.createOperation(returnState);
      
      return funcOp;
    }

    FuncOp NodeToFun::buildCoreFun(InstOp instOp) {
      Operation *calleeNodePtr = instOp.getCalleeNode();
      NodeOp nodeOp = dyn_cast<NodeOp>(calleeNodePtr);
      return buildCoreFun(nodeOp);
    }

    FuncOp NodeToFun::buildCoreFun(NodeOp nodeOp) {
      OpBuilder builder(nodeOp);
      builder.setInsertionPointAfter(nodeOp);

      SmallVector<Type, 4> neoInputs;
      for (Value i : nodeOp.getInputs()) {
	SiginType st = i.getType().cast<SiginType>();
	Type t = st.getType();
	FunctionType ft = BufferGenerator::inputFunction(builder, t);
	neoInputs.push_back(ft);
      }
      SmallVector<Type, 4> neoOutputs;
      for (Value i : nodeOp.getOutputs()) {
	SigoutType st = i.getType().cast<SigoutType>();
	Type t = st.getType();
	FunctionType ft = BufferGenerator::outputFunction(builder, t);
	neoOutputs.push_back(ft);
      }
      
      SmallVector<Type, 4> inputsOutputs;
      inputsOutputs.append({builder.getI32Type()});
      nodeOp.getBody().insertArgument((unsigned)0, builder.getI32Type());
      inputsOutputs.append(neoInputs.begin(), neoInputs.end());
      inputsOutputs.append(neoOutputs.begin(), neoOutputs.end());
      FunctionType ft = builder.getFunctionType(inputsOutputs, {});

      FuncOp funcOp = extFunctionPool->build(nodeOp.getNodeName().str(), ft);

      funcOp.getBody().takeBody(nodeOp.getBody());

      for (unsigned i = 1 ; i < inputsOutputs.size() ; i++) {
	Type t = inputsOutputs[i];
	Value neoArg = funcOp.getBody().insertArgument(i, t);
	Value oldArg = funcOp.getBody().getArgument(i + 1);
	oldArg.replaceAllUsesWith(neoArg);
	funcOp.getBody().eraseArgument(i + 1);
      }
      lowerIO(funcOp, funcOp,
	      nodeOp.getNumInputs() + nodeOp.getNumStatic());
      dishes.push_back(nodeOp);
      instances.insert(nodeOp);
      return funcOp;
    }

    void NodeToFun::lowerIO(FuncOp funcOp,Operation *op,unsigned offset_out) {
      
      for (Region &r: op->getRegions()) {
	for (Operation &deeperOp: *r.begin()) {
	  lowerIO(funcOp, &deeperOp, offset_out);
	}
      }

      ConstantPool cp(funcOp);

      if (isa<InputOp>(op)) {
	InputOp inputOp = dyn_cast<InputOp>(op);
	Operation *callOpPtr = lowerAndBufferizeInput(inputOp, funcOp, cp);
	inputOp.replaceAllUsesWith(callOpPtr);
	dishes.push_back(inputOp);
      }
      else if (isa<OutputOp>(op)) {
	OutputOp outputOp = dyn_cast<OutputOp>(op);
	Operation *callOpPtr = lowerAndBufferizeOutput(outputOp, funcOp, cp,
						       offset_out);
	outputOp.replaceAllUsesWith(callOpPtr);
	dishes.push_back(outputOp);
      }
    }

    Operation* NodeToFun::lowerAndBufferizeInput(InputOp inputOp,
						 FuncOp funcOp,
						 ConstantPool &cp) {
      OpBuilder builder(inputOp);
      Location loc = inputOp.getLoc();
      Value myFun = inputOp.getSignal();
      unsigned i = 0;
      for (Value arg : funcOp.getArguments().drop_front()) {
	if (arg == myFun) break;
	i++;
      }
      Value position = cp.buildInt(builder.getI32Type(), i);
      Type t = inputOp.getResult().getType();
      Value memRef = BufferGenerator::allocateType(builder, loc, t);
      OperationState callState(loc,
			       CallIndirectOp::getOperationName());
      CallIndirectOp::build(builder, callState, myFun,
			    {position, memRef});
      builder.createOperation(callState);
      Value myInput = BufferGenerator::unbufferize(builder, loc,
						   cp, t, memRef);
      return myInput.getDefiningOp();
    }

    Operation* NodeToFun::lowerAndBufferizeOutput(OutputOp outputOp,
						  FuncOp funcOp,
						  ConstantPool &cp,
						  unsigned offset_out) {
      OpBuilder builder(outputOp);
      Location loc = outputOp.getLoc();
      Value myFun = outputOp.getSignal();
      unsigned i = 0;
      for (Value arg : funcOp.getArguments().drop_front(offset_out + 1)) {
	if (arg == myFun) break;
	i++;
      }
      Value position = cp.buildInt(builder.getI32Type(), i);
      Value raw = outputOp.getParameter();
      Value param = BufferGenerator::bufferize(builder, loc, cp, raw);
      OperationState callState(loc,
			       CallIndirectOp::getOperationName());
      CallIndirectOp::build(builder, callState,
			    myFun,
			    {position, param});
      Operation *myCall = builder.createOperation(callState);
      return myCall;
    }

    void NodeToFun::instOpToCall(InstOp instOp, FuncOp toBeCalled) {
      OpBuilder builder(instOp);
      Location loc = instOp.getLoc();
      ConstantPool cpInst(instOp.getOperation()->getParentRegion());
      
      SmallVector<Value, 4> params;

      // Id
      
      Value instIdV = cpInst.buildInt(builder.getI32Type(), instOp.getId());
      params.push_back(instIdV);

      // Bufferize inputs

      for (Value v : instOp.getArgOperands()) {
	Value memRef = BufferGenerator::bufferize(builder, loc, cpInst, v);
	params.push_back(memRef);
      }

      // Bufferize outputs

      SmallVector<Value, 4> refinedOutputs;
      for (Type t : instOp.getResults().getTypes()) {
	Value memRef = BufferGenerator::allocateType(builder, loc, t);
	params.push_back(memRef);
	refinedOutputs.push_back(memRef);
      }

      // Call fun

      OperationState callState(loc, CallOp::getOperationName());
      CallOp::build(builder, callState, toBeCalled, params);
      builder.createOperation(callState);

      // tensorize buffers
      
      for (unsigned i = 0; i < instOp.getResults().size(); i++) {
	Value rawOut = instOp.getResult(i);
	Value refinedOut = refinedOutputs[i];
	Value finalOut = BufferGenerator::unbufferize(builder, loc, cpInst,
						      rawOut.getType(),
						      refinedOut);
	rawOut.replaceAllUsesWith(finalOut);
      }
    }
  }
}
