#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // Add Linalg dialect
#include "mlir/Dialect/SCF/IR/SCF.h"        // Add SCF dialect
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "sample/Transforms/Passes.h"

// Sample type check pipeline
void samplePassPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::sample::createSampleTypeCheckPass());
  pm.addPass(mlir::createCanonicalizerPass()); 
  pm.addPass(mlir::createSCCPPass());   
}

// MatMul optimization pipeline
void matmulOptPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::sample::createMatmulOptPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

int main(int argc, char *argv[]) {
  // Register our custom passes
  mlir::sample::registerPasses(); 

  // Create a dialect registry with all necessary dialects
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, 
                 mlir::arith::ArithDialect,
                 mlir::linalg::LinalgDialect,  // Add Linalg dialect
                 mlir::scf::SCFDialect>();     // Add SCF dialect
                 
  // Create context with the registry
  mlir::MLIRContext context(registry);

  // Register pass pipelines
  mlir::PassPipelineRegistration<> typeCheckPipeline(
      "sample-typecheck", "Checks the types of the func.FuncOp arguments",
      samplePassPipeline);
      
  mlir::PassPipelineRegistration<> matmulPipeline(
      "matmul-opt-pipeline", "Optimizes matrix multiplications with tiling",
      matmulOptPipeline);

  // Run the optimization tool
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "sample-opt", registry));
}