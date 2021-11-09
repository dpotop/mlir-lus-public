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

HEADERS = \
	Passes.h \
	PermuteLoops.h \

CFILES = \
	main.cpp \
	LoopPermutationPrimePass.cpp \
	BufferizeReshapePass.cpp \
	PermuteLoops.cpp \
	RemoveCopyPrimePass.cpp \
	PrimeLinalgToAffineLoops.cpp \

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

all: mlir-prime #externc.so

# In the following rule, -Y means that all standard includes are
# discarded, so that only the current folder is considered.
# Then, the absent argument after the first "--" means I
# provide no CFLAGS, so that makedepend cannot infer folders
# from it. Finally, I give the set of files.
# Rebuilding dependencies requires a clean slate.
depend: clean
	makedepend -f makefile.depend -Y --  -- $(CFILES)

mlir-prime : $(OBJFILES)
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
