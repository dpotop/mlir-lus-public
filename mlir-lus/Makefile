# export ASAN_SYMBOLIZER_PATH=/home/hugo/llvm/bin/llvm-symbolizer

include Makefile.config


CFLAGS =  \
	-fno-rtti -fvisibility-inlines-hidden \
	-Wall \
	-I $(LLVM)/include -I./include \
	-O3 \
	-g \
#	-fsanitize=address
#	-pedantic \

# rep de chaque header et de chaque cfile
# soit rajouter un -I pour chacun des reps, soit modifier le chemin relatif

DIAL_DIR = Dialects/
DIAL_LUS_DIR = $(DIAL_DIR)Lus/
DIAL_PSSAO_DIR = $(DIAL_DIR)Pssa/
DIAL_SYNC_DIR = $(DIAL_DIR)Sync/

TRANS_DIR = Transforms/
TRANS_LUS_LUS_DIR = $(TRANS_DIR)LusToLus/
TRANS_LUS_PSSA_DIR = $(TRANS_DIR)LusToPssa/
TRANS_LUS_STD_DIR = $(TRANS_DIR)LusToStd/
TRANS_PSSA_PSSA_DIR = $(TRANS_DIR)PssaToPssa/
TRANS_SYNC_STD_DIR = $(TRANS_DIR)SyncToStd/
TRANS_LUS_SYNC_DIR = $(TRANS_DIR)LusToSync/
TRANS_PSSA_SYNC_DIR = $(TRANS_DIR)PssaToSync/
TRANS_PSSA_STD_DIR = $(TRANS_DIR)PssaToStd/
TRANS_UTILITIES_DIR = $(TRANS_DIR)Utilities/

PASSES_DIR = $(TRANS_DIR)Passes/

TOOLS_DIR = Tools/

H_DIAL_LUS = \
	$(DIAL_LUS_DIR)Clock.h \
	$(DIAL_LUS_DIR)ClockAnalysis.h \
	$(DIAL_LUS_DIR)ClockTree.h \
	$(DIAL_LUS_DIR)FbyOp.h \
	$(DIAL_LUS_DIR)PreOp.h \
	$(DIAL_LUS_DIR)Instance.h \
	$(DIAL_LUS_DIR)KPeriodic.h \
	$(DIAL_LUS_DIR)lus.h \
	$(DIAL_LUS_DIR)Node.h \
	$(DIAL_LUS_DIR)NodeType.h \
	$(DIAL_LUS_DIR)TestCondition.h \
	$(DIAL_LUS_DIR)ValueHash.h \
	$(DIAL_LUS_DIR)WhenOp.h \
	$(DIAL_LUS_DIR)MergeOp.h \
	$(DIAL_LUS_DIR)WhenType.h \
	$(DIAL_LUS_DIR)Yield.h \
	$(DIAL_LUS_DIR)YieldType.h \
	$(DIAL_LUS_DIR)SignalTypes.h \

CPP_DIAL_LUS = \
	$(DIAL_LUS_DIR)Clock.cpp \
	$(DIAL_LUS_DIR)ClockAnalysis.cpp \
	$(DIAL_LUS_DIR)ClockTree.cpp \
	$(DIAL_LUS_DIR)FbyOp.cpp \
	$(DIAL_LUS_DIR)PreOp.cpp \
	$(DIAL_LUS_DIR)Instance.cpp \
	$(DIAL_LUS_DIR)KPeriodic.cpp \
	$(DIAL_LUS_DIR)lus.cpp \
	$(DIAL_LUS_DIR)Node.cpp \
	$(DIAL_LUS_DIR)TestCondition.cpp \
	$(DIAL_LUS_DIR)WhenOp.cpp \
	$(DIAL_LUS_DIR)MergeOp.cpp \
	$(DIAL_LUS_DIR)Yield.cpp \

H_DIAL_PSSAO = \
	$(DIAL_PSSAO_DIR)CondactOp.h \
	$(DIAL_PSSAO_DIR)YieldOp.h \
	$(DIAL_PSSAO_DIR)CreatePredOp.h \
	$(DIAL_PSSAO_DIR)pssa.h \
	$(DIAL_PSSAO_DIR)CreatePredType.h \
	$(DIAL_PSSAO_DIR)OutputOp.h \

CPP_DIAL_PSSAO = \
	$(DIAL_PSSAO_DIR)CondactOp.cpp \
	$(DIAL_PSSAO_DIR)YieldOp.cpp \
	$(DIAL_PSSAO_DIR)CreatePredOp.cpp \
	$(DIAL_PSSAO_DIR)pssa.cpp \
	$(DIAL_PSSAO_DIR)OutputOp.cpp \

H_DIAL_SYNC = \
	$(DIAL_SYNC_DIR)SignalTypes.h \
	$(DIAL_SYNC_DIR)NodeType.h \
	$(DIAL_SYNC_DIR)Node.h \
	$(DIAL_SYNC_DIR)Sync.h \
	$(DIAL_SYNC_DIR)HaltOp.h \
	$(DIAL_SYNC_DIR)InputOp.h \
	$(DIAL_SYNC_DIR)OutputOp.h \
	$(DIAL_SYNC_DIR)TickOp.h \
	$(DIAL_SYNC_DIR)SyncOp.h \
	$(DIAL_SYNC_DIR)InOp.h \
	$(DIAL_SYNC_DIR)OutOp.h \
	$(DIAL_SYNC_DIR)InstOp.h \
	$(DIAL_SYNC_DIR)UndefOp.h \
	$(DIAL_SYNC_DIR)SelectOp.h \

CPP_DIAL_SYNC = \
	$(DIAL_SYNC_DIR)Sync.cpp \
	$(DIAL_SYNC_DIR)Node.cpp \
	$(DIAL_SYNC_DIR)InOp.cpp \
	$(DIAL_SYNC_DIR)InputOp.cpp \
	$(DIAL_SYNC_DIR)InstOp.cpp \
	$(DIAL_SYNC_DIR)OutOp.cpp \
	$(DIAL_SYNC_DIR)OutputOp.cpp \
	$(DIAL_SYNC_DIR)SyncOp.cpp \
	$(DIAL_SYNC_DIR)TickOp.cpp \
	$(DIAL_SYNC_DIR)SelectOp.cpp \

H_DIAL = $(H_DIAL_LUS) $(H_DIAL_PSSAO) $(H_DIAL_SYNC) $(DIAL_DIR)MinMaxOperands.h
CPP_DIAL = $(CPP_DIAL_LUS) $(CPP_DIAL_PSSAO) $(CPP_DIAL_SYNC)

H_TOOLS = \
	$(TOOLS_DIR)ParserAux.h \
	$(TOOLS_DIR)CommandLine.h \

CPP_TOOLS = \
	$(TOOLS_DIR)main.cpp \
	$(TOOLS_DIR)ParserAux.cpp \
	$(TOOLS_DIR)CommandLine.cpp \

H_TRANS_LUS_LUS = \
	$(TRANS_LUS_LUS_DIR)EnsureDominance.h \
	$(TRANS_LUS_LUS_DIR)InlineNodes.h \
	$(TRANS_LUS_LUS_DIR)RemoveFby.h \
	$(TRANS_LUS_LUS_DIR)PersistFby.h \
	$(TRANS_LUS_LUS_DIR)RemovePre.h \

CPP_TRANS_LUS_LUS = \
	$(TRANS_LUS_LUS_DIR)EnsureDominance.cpp \
	$(TRANS_LUS_LUS_DIR)InlineNodes.cpp \
	$(TRANS_LUS_LUS_DIR)RemovePre.cpp \
	$(TRANS_LUS_LUS_DIR)RemoveFby.cpp \
	$(TRANS_LUS_LUS_DIR)PersistFby.cpp \

H_TRANS_SYNC_STD = \
	$(TRANS_SYNC_STD_DIR)NodeToFun.h \
	$(TRANS_SYNC_STD_DIR)LowerTick.h \
	$(TRANS_SYNC_STD_DIR)RemovePsis.h \
	$(TRANS_SYNC_STD_DIR)RemoveSomeUndefs.h \

CPP_TRANS_SYNC_STD = \
	$(TRANS_SYNC_STD_DIR)NodeToFun.cpp \
	$(TRANS_SYNC_STD_DIR)LowerTick.cpp \
	$(TRANS_SYNC_STD_DIR)SyncToStdPass.cpp \
	$(TRANS_SYNC_STD_DIR)RemovePsis.cpp \
	$(TRANS_SYNC_STD_DIR)RemoveSomeUndefs.cpp \

H_TRANS_LUS_SYNC = \
	$(TRANS_LUS_SYNC_DIR)NodeToNode.h \
	$(TRANS_LUS_SYNC_DIR)GenCondacts.h \
	$(TRANS_LUS_SYNC_DIR)GenPredicates.h \
	$(TRANS_LUS_SYNC_DIR)LowerPssa.h \
	$(TRANS_LUS_SYNC_DIR)FusionCondacts.h \
	$(TRANS_LUS_SYNC_DIR)LowerInst.h \
	$(TRANS_LUS_SYNC_DIR)GenOutputs.h \

CPP_TRANS_LUS_SYNC = \
	$(TRANS_LUS_SYNC_DIR)NodeToNode.cpp \
	$(TRANS_LUS_SYNC_DIR)GenCondacts.cpp \
	$(TRANS_LUS_SYNC_DIR)GenPredicates.cpp \
	$(TRANS_LUS_SYNC_DIR)LowerPssa.cpp \
	$(TRANS_LUS_SYNC_DIR)LusToSyncPass.cpp \
	$(TRANS_LUS_SYNC_DIR)PssaToStdPass.cpp \
	$(TRANS_LUS_SYNC_DIR)NodeToNodePass.cpp \
	$(TRANS_LUS_SYNC_DIR)FusionCondacts.cpp \
	$(TRANS_LUS_SYNC_DIR)LowerInst.cpp \
	$(TRANS_LUS_SYNC_DIR)GenOutputs.cpp \

H_TRANS_UTILITIES = \
	$(TRANS_UTILITIES_DIR)CondToPred.h \
	$(TRANS_UTILITIES_DIR)ConstantPool.h \
	$(TRANS_UTILITIES_DIR)OperationsAux.h \
	$(TRANS_UTILITIES_DIR)ExtFunctionPool.h \
	$(TRANS_UTILITIES_DIR)BufferGenerator.h \

CPP_TRANS_UTILITIES = \
	$(TRANS_UTILITIES_DIR)OperationsAux.cpp \
	$(TRANS_UTILITIES_DIR)ConstantPool.cpp \
	$(TRANS_UTILITIES_DIR)ExtFunctionPool.cpp \
	$(TRANS_UTILITIES_DIR)BufferGenerator.cpp \

H_PASSES = \
	$(PASSES_DIR)Passes.h \

CPP_PASSES = \
	$(PASSES_DIR)UpOpsPass.cpp \
	$(PASSES_DIR)SyncToLLVMPass.cpp \
	$(PASSES_DIR)RemoveFbyPass.cpp \
	$(PASSES_DIR)PersistFbyPass.cpp \
	$(PASSES_DIR)RemovePrePass.cpp \
	$(PASSES_DIR)ScheduleDominancePass.cpp \
	$(PASSES_DIR)GenPredicatesPass.cpp \
	$(PASSES_DIR)GenCondactsPass.cpp \
	$(PASSES_DIR)FusionCondactsPass.cpp \
	$(PASSES_DIR)InlineNodesPass.cpp \
	$(PASSES_DIR)GenOutputsPass.cpp \

H_TRANS = \
	$(H_PASSES) \
	$(H_TRANS_LUS_LUS) \
	$(H_TRANS_UTILITIES) \
	$(H_TRANS_SYNC_STD) \
	$(H_TRANS_LUS_SYNC) \

CPP_TRANS = \
	$(CPP_PASSES) \
	$(CPP_TRANS_LUS_LUS) \
	$(CPP_TRANS_UTILITIES) \
	$(CPP_TRANS_SYNC_STD) \
	$(CPP_TRANS_LUS_SYNC) \

HEADERS = $(H_DIAL) $(H_TOOLS) $(H_TRANS)

CFILES = $(CPP_DIAL) $(CPP_TOOLS) $(CPP_TRANS)

OBJFILES = $(CFILES:.cpp=.o)

LIBS = -lExampleIRTransforms \
-lLLVMAArch64AsmParser \
-lLLVMAArch64CodeGen \
-lLLVMAArch64Desc \
-lLLVMAArch64Disassembler \
-lLLVMAArch64Info \
-lLLVMAArch64Utils \
-lLLVMAggressiveInstCombine \
-lLLVMAnalysis \
-lLLVMAsmParser \
-lLLVMAsmPrinter \
-lLLVMBinaryFormat \
-lLLVMBitReader \
-lLLVMBitstreamReader \
-lLLVMBitWriter \
-lLLVMCFGuard \
-lLLVMCFIVerify \
-lLLVMCodeGen \
-lLLVMCore \
-lLLVMCoroutines \
-lLLVMCoverage \
-lLLVMDebugInfoCodeView \
-lLLVMDebugInfoDWARF \
-lLLVMDebugInfoGSYM \
-lLLVMDebugInfoMSF \
-lLLVMDebugInfoPDB \
-lLLVMDemangle \
-lLLVMDlltoolDriver \
-lLLVMDWARFLinker \
-lLLVMExecutionEngine \
-lLLVMExegesis \
-lLLVMExegesisAArch64 \
-lLLVMExtensions \
-lLLVMFileCheck \
-lLLVMFrontendOpenACC \
-lLLVMFrontendOpenMP \
-lLLVMFuzzMutate \
-lLLVMGlobalISel \
-lLLVMInstCombine \
-lLLVMInstrumentation \
-lLLVMInterfaceStub \
-lLLVMInterpreter \
-lLLVMipo \
-lLLVMIRReader \
-lLLVMJITLink \
-lLLVMLibDriver \
-lLLVMLineEditor \
-lLLVMLinker \
-lLLVMLTO \
-lLLVMMC \
-lLLVMMCA \
-lLLVMMCDisassembler \
-lLLVMMCJIT \
-lLLVMMCParser \
-lLLVMMIRParser \
-lLLVMObjCARCOpts \
-lLLVMObject \
-lLLVMObjectYAML \
-lLLVMOption \
-lLLVMOrcJIT \
-lLLVMOrcShared \
-lLLVMOrcTargetProcess \
-lLLVMPasses \
-lLLVMProfileData \
-lLLVMRemarks \
-lLLVMRuntimeDyld \
-lLLVMScalarOpts \
-lLLVMSelectionDAG \
-lLLVMSupport \
-lLLVMSymbolize \
-lLLVMTableGen \
-lLLVMTableGenGlobalISel \
-lLLVMTarget \
-lLLVMTextAPI \
-lLLVMTransformUtils \
-lLLVMVectorize \
-lLLVMWindowsManifest \
-lLLVMXRay \
-lMLIRAffine \
-lMLIRAffineEDSC \
-lMLIRAffineToStandard \
-lMLIRAffineTransforms \
-lMLIRAffineTransformsTestPasses \
-lMLIRAffineUtils \
-lMLIRAMX \
-lMLIRAMXToLLVMIRTranslation \
-lMLIRAMXTransforms \
-lMLIRAnalysis \
-lMLIRArmNeon \
-lMLIRArmNeonToLLVMIRTranslation \
-lMLIRArmSVE \
-lMLIRArmSVEToLLVM \
-lMLIRAsync \
-lMLIRAsyncToLLVM \
-lMLIRAsyncTransforms \
-lMLIRAVX512 \
-lMLIRAVX512ToLLVMIRTranslation \
-lMLIRAVX512Transforms \
-lMLIRCallInterfaces \
-lMLIRCAPIConversion \
-lMLIRCAPIIR \
-lMLIRCAPILinalg \
-lMLIRCAPIRegistration \
-lMLIRCAPISCF \
-lMLIRCAPIShape \
-lMLIRCAPIStandard \
-lMLIRCAPITensor \
-lMLIRCAPITransforms \
-lMLIRCastInterfaces \
-lMLIRCEXECUTIONENGINE \
-lMLIRComplex \
-lMLIRComplexToLLVM \
-lMLIRControlFlowInterfaces \
-lMLIRCopyOpInterface \
-lMLIRDataLayoutInterfaces \
-lMLIRDerivedAttributeOpInterface \
-lMLIRDialect \
-lMLIRDialectUtils \
-lMLIRDLTI \
-lMLIREDSC \
-lMLIRExecutionEngine \
-lMLIRGPU \
-lMLIRGPUToGPURuntimeTransforms \
-lMLIRGPUToNVVMTransforms \
-lMLIRGPUToROCDLTransforms \
-lMLIRGPUToSPIRV \
-lMLIRGPUToVulkanTransforms \
-lMLIRInferTypeOpInterface \
-lMLIRIR \
-lMLIRJitRunner \
-lMLIRLinalg \
-lMLIRLinalgAnalysis \
-lMLIRLinalgEDSC \
-lMLIRLinalgToLLVM \
-lMLIRLinalgToSPIRV \
-lMLIRLinalgToStandard \
-lMLIRLinalgTransforms \
-lMLIRLinalgUtils \
-lMLIRLLVMArmSVE \
-lMLIRLLVMArmSVEToLLVMIRTranslation \
-lMLIRLLVMIR \
-lMLIRLLVMIRTransforms \
-lMLIRLLVMToLLVMIRTranslation \
-lMLIRLoopAnalysis \
-lMLIRLoopLikeInterface \
-lMLIRMath \
-lMLIRMathTransforms \
-lMLIRMemRef \
-lMLIRMlirOptMain \
-lMLIRNVVMIR \
-lMLIRNVVMToLLVMIRTranslation \
-lMLIROpenACC \
-lMLIROpenMP \
-lMLIROpenMPToLLVM \
-lMLIROpenMPToLLVMIRTranslation \
-lMLIROptLib \
-lMLIRParser \
-lMLIRPass \
-lMLIRPDL \
-lMLIRPDLInterp \
-lMLIRPDLToPDLInterp \
-lMLIRPresburger \
-lMLIRQuant \
-lMLIRReduce \
-lMLIRRewrite \
-lMLIRROCDLIR \
-lMLIRROCDLToLLVMIRTranslation \
-lMLIRSCF \
-lMLIRSCFToGPU \
-lMLIRSCFToOpenMP \
-lMLIRSCFToSPIRV \
-lMLIRSCFToStandard \
-lMLIRSCFTransforms \
-lMLIRSDBM \
-lMLIRShape \
-lMLIRShapeOpsTransforms \
-lMLIRShapeTestPasses \
-lMLIRShapeToStandard \
-lMLIRSideEffectInterfaces \
-lMLIRSPIRV \
-lMLIRSPIRVBinaryUtils \
-lMLIRSPIRVConversion \
-lMLIRSPIRVDeserialization \
-lMLIRSPIRVModuleCombiner \
-lMLIRSPIRVSerialization \
-lMLIRSPIRVTestPasses \
-lMLIRSPIRVToLLVM \
-lMLIRSPIRVTransforms \
-lMLIRSPIRVTranslateRegistration \
-lMLIRSPIRVUtils \
-lMLIRStandard \
-lMLIRStandardOpsTransforms \
-lMLIRStandardToLLVM \
-lMLIRStandardToSPIRV \
-lMLIRSupport \
-lMLIRSupportIndentedOstream \
-lMLIRTableGen \
-lMLIRTargetLLVMIRExport \
-lMLIRTargetLLVMIRImport \
-lMLIRTensor \
-lMLIRTensorTransforms \
-lMLIRTestAnalysis \
-lMLIRTestDialect \
-lMLIRTestIR \
-lMLIRTestPass \
-lMLIRTestReducer \
-lMLIRTestRewrite \
-lMLIRTestTransforms \
-lMLIRToLLVMIRTranslationRegistration \
-lMLIRTosa \
-lMLIRTosaTestPasses \
-lMLIRTosaToLinalg \
-lMLIRTosaToSCF \
-lMLIRTosaToStandard \
-lMLIRTosaTransforms \
-lMLIRTransforms \
-lMLIRTransformUtils \
-lMLIRTranslation \
-lMLIRVector \
-lMLIRVectorInterfaces \
-lMLIRVectorToLLVM \
-lMLIRVectorToROCDL \
-lMLIRVectorToSCF \
-lMLIRVectorToSPIRV \
-lMLIRViewLikeInterface \

all: mlirlus #externc.so

# In the following rule, -Y means that all standard includes are
# discarded, so that only the current folder is considered.
# Then, the absent argument after the first "--" means I
# provide no CFLAGS, so that makedepend cannot infer folders
# from it. Finally, I give the set of files.
# Rebuilding dependencies requires a clean slate.
depend: clean
	makedepend -f makefile.depend --Y  -- $(CFILES)

mlirlus : $(OBJFILES)
	@echo "Linking..."
	@$(LD) $(LDFLAGS) -o $@ $(OBJFILES) $(LIBS) $(LIBS) $(LIBS) $(LIBS) $(LIBS) -lz -lncurses

#externc.so: externc.c
#	$(CPP) -shared $< -o $@

%.o: %.cpp
	@echo "Compiling $<..."
	@$(CPP) $(CFLAGS) -c $< -o $@

clean:
	@rm -f $(OBJFILES) *~ main externc.so
	@rm -f *.bak

include makefile.depend
# DO NOT DELETE
