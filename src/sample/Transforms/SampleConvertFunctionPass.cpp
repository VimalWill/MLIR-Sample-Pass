#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "sample/Transforms/Passes.h"

namespace mlir::sample {
#define GEN_PASS_DEF_SAMPLECONVERTFUNCTION
#include "sample/Transforms/Passes.h.inc"

namespace {
class SampleConvertFunctionPass
    : public impl::SampleConvertFunctionBase<SampleConvertFunctionPass> {
public:
  void runOnOperation() override;
};
} // namespace

LogicalResult maybeAndRewrite(func::FuncOp Op, IRRewriter &rewriter) {
  auto funcName = Op.getSymName();
  auto newFuncName = (funcName + "_converted").str();
  Op.setVisibility(SymbolTable::Visibility::Private);

  MLIRContext *ctx = Op->getContext();
  FunctionType newFuncType = FunctionType::get(ctx, {}, {});

  rewriter.setInsertionPoint(Op);
  auto newFuncOp =
      rewriter.create<func::FuncOp>(Op.getLoc(), newFuncName, newFuncType);

  if (!newFuncOp) {
    llvm::errs() << "unable to create new func-op\n";
    return failure();
  }

  auto entryBlock = newFuncOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  auto i32Type = rewriter.getI32Type();
  auto valueAttr = rewriter.getI32IntegerAttr(42);

  rewriter.create<arith::ConstantOp>(newFuncOp.getLoc(), i32Type, valueAttr);
  rewriter.create<func::ReturnOp>(newFuncOp.getLoc());

  rewriter.eraseOp(Op);
  return success();
}

void SampleConvertFunctionPass::runOnOperation() {
  auto funcOp = dyn_cast<func::FuncOp>(getOperation());
  if (!funcOp)
    return;

  MLIRContext *ctx = &getContext();
  IRRewriter rewriter(ctx);

  if (failed(maybeAndRewrite(funcOp, rewriter))) {
    funcOp.emitError("failed to convert function operation");
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> createSampleConvertFunctionPass() {
  return std::make_unique<SampleConvertFunctionPass>();
}
} // namespace mlir::sample
