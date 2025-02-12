#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "sample/Transforms/Passes.h"

void samplePassPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::sample::createSampleTypeCheckPass());
  pm.addPass(mlir::createCanonicalizerPass()); 
  pm.addPass(mlir::createSCCPPass());   
}

int main(int argc, char *argv[]) {
  mlir::sample::registerPasses(); 

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);

  mlir::PassPipelineRegistration<> pipeline(
      "sample-typecheck", "Checks the types of the func.FuncOp arguments",
      samplePassPipeline);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "sample-opt", registry));
}
