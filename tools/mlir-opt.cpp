#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h" // Add Linalg dialect
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h" // Add SCF dialect
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

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

void sampleFuncPassPipeline(mlir::OpPassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::sample::createSampleConvertFunctionPass());
  // pm.addPass(mlir::createCanonicalizerPass());
  // pm.addPass(mlir::createSCCPPass());
}

int main(int argc, char *argv[]) {
  // Register our custom passes
  mlir::sample::registerPasses();

  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::BuiltinDialect, // ← This is what you're likely missing
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::tosa::TosaDialect, mlir::memref::MemRefDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects(); // optional but helpful

  // Register pass pipelines
  mlir::PassPipelineRegistration<> typeCheckPipeline(
      "sample-typecheck", "Checks the types of the func.FuncOp arguments",
      samplePassPipeline);

  mlir::PassPipelineRegistration<> matmulPipeline(
      "matmul-opt-pipeline", "Optimizes matrix multiplications with tiling",
      matmulOptPipeline);

  mlir::PassPipelineRegistration<> funcConvertPipeline(
      "sample-convertfun", "converts the function signature",
      sampleFuncPassPipeline);

  // Run the optimization tool
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "sample-opt", registry));
}