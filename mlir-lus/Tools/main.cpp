#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "../Dialects/Lus/lus.h"
#include "../Dialects/Pssa/pssa.h"
#include "../Dialects/Sync/Sync.h"
#include "../Dialects/Sync/Node.h"
#include "../Dialects/Lus/Node.h"
#include "../Transforms/Passes/Passes.h"
#include "../Tools/CommandLine.h"
#include "mlir/Dialect/SCF/SCF.h"

using namespace mlir ;
using namespace mlir::vector;

//=========================================================
// Accepted command line arguments:

//---------------------------------------------------------
// input and output files
static llvm::cl::opt<std::string>
inputFilename(llvm::cl::Positional,
	      llvm::cl::desc("First positional argument, it sets the name of the input file. By default, it is -, which means input is taken from standard input."),
	      llvm::cl::init("-"));
static llvm::cl::opt<std::string>
outputFilename("o",
	       llvm::cl::desc("Set output filename"),
	       llvm::cl::value_desc("filename"),
	       llvm::cl::init("-"));

//---------------------------------------------------------
// Processing pipeline control
static llvm::cl::opt<bool>
inlineNodes("inline-nodes",
	    llvm::cl::desc("Inlines all node instances that are not marked noinline."),
	    llvm::cl::init(false));
static llvm::cl::opt<bool>
ensureLusDom("normalize",
	     llvm::cl::desc("Normalize lus nodes of an MLIR file."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
predicateLus("predicate",
	     llvm::cl::desc("Adds pssa dialect predication to a lus node. This makes predication (clocks) explicit."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
nodeToFun("convert-lus-to-std",
	  llvm::cl::desc("Lowers lus operations (including nodes) to the standard dialect. Assumes predicates have already been synthesized."),
	  llvm::cl::init(false));
static llvm::cl::opt<bool>
lusToSync("convert-lus-to-sync",
	  llvm::cl::desc("Lowers lus operations (including nodes) to the sync dialect."),
	  llvm::cl::init(false));
static llvm::cl::opt<std::string>
futureMain("mainnode",
	   llvm::cl::desc("Upon non-modular lowering of lus nodes to std functions, sets the name of the node that will be lowered."),
	   llvm::cl::value_desc("A string or -"),
	   llvm::cl::init("-"));
static llvm::cl::opt<bool>
lowerPssa("convert-pssa-to-std",
	  llvm::cl::desc("Lower pssa operations and types to the standard dialect. Assumes lus constructs have already been lowered to the std dialect."),
	  llvm::cl::init(false));
static llvm::cl::opt<bool>
lowerPssaToLLVM("convert-pssa-to-llvm",
		llvm::cl::desc("Lower pssa operations and types to the LLVM dialect."),
		llvm::cl::init(false));
static llvm::cl::opt<bool>
lowerSyncToStd("convert-sync-to-std",
	  llvm::cl::desc("Lower sync operations and types to the standard dialect."),
	  llvm::cl::init(false));
static llvm::cl::opt<bool>
lowerSyncToLLVM("convert-sync-to-llvm",
	       llvm::cl::desc("Lower sync operations and types to the LLVM dialect."),
	       llvm::cl::init(false));
static llvm::cl::opt<bool>
upOps("up-ops",
	       llvm::cl::desc("Up ops."),
	       llvm::cl::init(false));

//---------------------------------------------------------
// Debugging options
static llvm::cl::opt<unsigned>
verbose("verbose",
	llvm::cl::desc("Print more tracing information in all stages. The integer argument is the verbosity level. Zero means no info."),
	llvm::cl::init(0));
static llvm::cl::opt<bool>
disableCA("disable-clock-analysis",
	  llvm::cl::desc("Disable clock analysis. As code generation depends on clock analysis, this can only be used in order to debug the parser."),
	  llvm::cl::init(false));
static llvm::cl::opt<bool>
showDialects("show-dialects",
	     llvm::cl::desc("Print the list of registered dialects and exit."),
	     llvm::cl::init(false));



//=========================================================
//
// This function may be called to register the MLIR passes with the
// global registry.  If you're building a compiler, you likely don't
// need this: you would build a pipeline programmatically without the
// need to register with the global registry, since it would already
// be calling the creation routine of the individual passes.  The
// global registry is interesting to interact with the command-line
// tools.
void registerAllPassesSpecialized() {
  // Init general passes
  createCanonicalizerPass();
  createCSEPass();
  createLoopUnrollPass();
  createLoopUnrollAndJamPass();
  createSimplifyAffineStructuresPass();
  createLoopFusionPass();
  createLoopInvariantCodeMotionPass();
  createAffineLoopInvariantCodeMotionPass();
  createPipelineDataTransferPass();
  createLowerAffinePass();
  createLoopTilingPass(0);
  createLoopCoalescingPass();
  createAffineDataCopyGenerationPass(0, 0);  
  createStripDebugInfoPass();
  createInlinerPass();

  createSuperVectorizePass({});
  createMemRefDataFlowOptPass();
  createPrintOpStatsPass();
  createSymbolDCEPass();
  createLocationSnapshotPass({});
  
  // Linalg
  createLinalgFusionOfTensorOpsPass();
  createLinalgTilingPass();
  createLinalgPromotionPass(false,false);
  createConvertLinalgToLoopsPass();
  createConvertLinalgToAffineLoopsPass();

  createLinalgTilingToParallelLoopsPass();
  createConvertLinalgToParallelLoopsPass();
  createConvertLinalgToLLVMPass();

  // LoopOps
  createParallelLoopFusionPass();
  createParallelLoopSpecializationPass();
  createParallelLoopTilingPass();
}


//=========================================================
// Main driver
int main(int argc, char **argv) {
  
  //--------------------------------------------------
  // Parse command line. This must happen **after**
  // initialization of MLIR and LLVM, because init adds
  // options.
  mlir::registerPassManagerCLOptions();
  // The following line implements a command-line parser for
  // MLIR passes.
  PassPipelineCLParser passPipeline("", "Compiler passes to run");
  llvm::cl::ParseCommandLineOptions(argc,argv,
				    "Work-in-progress tool\n");

  verboseLevel = verbose ;
  if(disableCA) {
    // llvm::outs() << "Clock analysis disabled!\n";
    disableClockAnalysis = true ;
  }
  if (futureMain.getNumOccurrences() > 0) {
    hasMainNode = true;
    mainNode = futureMain.getValue();
  }


  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::lus::Lus>();
  context.getOrLoadDialect<mlir::pssa::Pssa>();
  context.getOrLoadDialect<mlir::sync::Sync>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::tosa::TosaDialect>();

  context.allowUnregisteredDialects();
  
  //--------------------------------------------------
  // Set up input and output files - must be done after
  // parsing the command line.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> ifile =
    openInputFile(inputFilename, &errorMessage);
  if (!ifile ) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  } 
  std::unique_ptr<llvm::ToolOutputFile> ofile =
    openOutputFile(outputFilename, &errorMessage);
  if (!ofile ) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  
 
  //--------------------------------------------------
  // Load the input file into the context
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr,&context);
  sourceMgr.AddNewSourceBuffer(std::move(ifile), llvm::SMLoc());
  OwningModuleRef module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return -2;
  }
 
  //--------------------------------------------------
  //
  {
    // Apply any pass manager command line options.
    mlir::PassManager pm(&context);
    applyPassManagerCLOptions(pm);

    if(inlineNodes) {
      pm.addPass(mlir::lus::createInlineNodesPass());
    }
    
    mlir::OpPassManager &lusNodePM = pm.nest<mlir::lus::NodeOp>();
    
    if(ensureLusDom) {
      lusNodePM.addPass(mlir::lus::createRemoveFbyPass());
      lusNodePM.addPass(mlir::lus::createRemovePrePass());
      lusNodePM.addPass(mlir::lus::createScheduleDominancePass());
      // lusNodePM.addPass(mlir::lus::createEnsureLusDominancePass());
    }
    if (predicateLus) {
      lusNodePM.addPass(mlir::lus::createGenPredicatesPass());
      lusNodePM.addPass(mlir::lus::createGenCondactsPass());
      lusNodePM.addPass(mlir::pssa::createFusionCondactsPass());
    }
    if (lusToSync) {
      lusNodePM.addPass(mlir::lus::createGenOutputsPass());
      lusNodePM.addPass(mlir::lus::createGenPredicatesPass());
      lusNodePM.addPass(mlir::lus::createGenCondactsPass());
      lusNodePM.addPass(mlir::pssa::createFusionCondactsPass());
      // lusNodePM.addPass(mlir::lus::createLusToPssaPass());
      pm.addPass(mlir::lus::createLusToSyncPass());
      // pm.addPass(mlir::pssa::createPssaToStandardPass());
    }

    if (lowerSyncToStd) {
           pm.addPass(mlir::sync::createSyncToStandardPass());
    }
    if (lowerSyncToLLVM) {
      pm.addPass(mlir::createLowerToCFGPass());
      pm.addPass(mlir::sync::createSyncToLLVMPass());
    }

    if (upOps) {
      pm.addPass(mlir::createUpOpsPass());
    }
    
    // Build the provided pipeline from standard command line
    // arguments
    function_ref<LogicalResult(const Twine &)> errorHandler;
    if (mlir::failed(passPipeline.addToPipeline(pm,errorHandler)))
      return -1 ;
    
    // Run the pipeline.
    if (mlir::failed(pm.run(*module)))
      return -2 ;
  }

  //--------------------------------------------------
  // Print the result in the output file. If no name is
  // provided, this file is the standard output. 
  // Note the use of the -> operator which is equivalent to
  // module.get().
  module->print(ofile->os()) ;
  ofile->os() << "\n" ;
  // Make sure the file is not deleted
  ofile->keep() ;

  // Normal return value
  return 0 ;
}
