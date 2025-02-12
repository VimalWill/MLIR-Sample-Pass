#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "sample/Transforms/Passes.h"

using namespace mlir;

namespace mlir::sample {

#define GEN_PASS_DEF_SAMPLETYPECHECK
#include "sample/Transforms/Passes.h.inc"

namespace {
class SampleTypeCheck : public impl::SampleTypeCheckBase<SampleTypeCheck> {
public:
  void runOnOperation() override;
};
} // namespace

static bool isValidArgType(Type argType) { return argType.isInteger(8); }

void SampleTypeCheck::runOnOperation() {
  ModuleOp module = dyn_cast<ModuleOp>(getOperation());
  bool hasError = false;

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    for (BlockArgument arg : funcOp.getArguments()) {
      if (!isValidArgType(arg.getType())) {
        funcOp.emitError() << "Invalid argument type: " << arg.getType()
                           << ". Expected i8.";
        hasError = true;
      }
    }
  }

  if (hasError) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> createSampleTypeCheckPass() {
  return std::make_unique<SampleTypeCheck>();
}
} // namespace mlir::sample
