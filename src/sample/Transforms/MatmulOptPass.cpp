#include "sample/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct LinalgMatmulTile : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::MatmulOp matmulOp, 
      PatternRewriter &rewriter) const override {
    llvm::errs() << "Found a matmulOp to transform!\n";
    
    // Tiling configuration
    SmallVector<int64_t> tileSizes{8, 8};
    auto options = linalg::LinalgTilingOptions()
      .setTileSizes(tileSizes)
      .setLoopType(linalg::LinalgTilingLoopType::Loops);

    llvm::errs() << "Attempting to tile matmulOp...\n";
    
    // Perform tiling
    FailureOr<linalg::TiledLinalgOp> tiled = 
      linalg::tileLinalgOp(rewriter, matmulOp, options);
    
    if (failed(tiled)) {
      llvm::errs() << "Tiling failed!\n";
      return failure();
    }
    
    llvm::errs() << "Tiling succeeded! Replacing original op.\n";
    rewriter.replaceOp(matmulOp, tiled->tensorResults);
    return success();
  }
};
} // end anonymous namespace

namespace mlir {
namespace sample {

class MatmulOptPass : public OperationPass<> {
public:
  MatmulOptPass() : OperationPass(TypeID::get<MatmulOptPass>()) {}

  StringRef getName() const override { return "MatmulOpt"; }
  StringRef getArgument() const override { return "matmul-opt"; }
  StringRef getDescription() const override { return "Optimize matrix multiplications using tiling"; }

  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<MatmulOptPass>();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    llvm::errs() << "MatmulOptPass is running!\n";
    Operation *op = getOperation();
    
    // Collect all matmul ops first
    SmallVector<linalg::MatmulOp, 4> matmulOps;
    op->walk([&](linalg::MatmulOp matmulOp) {
      matmulOps.push_back(matmulOp);
    });
    
    if (matmulOps.empty()) {
      llvm::errs() << "No linalg.matmul operations found!\n";
      return;
    }
    
    llvm::errs() << "Found " << matmulOps.size() << " linalg.matmul operations\n";
    
    // Process each matmul op individually
    bool anyFailed = false;
    for (auto matmulOp : matmulOps) {
      // Create a rewriter with the correct context pointer
      IRRewriter rewriter(&getContext());
      
      // Tiling configuration
      SmallVector<int64_t> tileSizes{8, 8};
      auto options = linalg::LinalgTilingOptions()
        .setTileSizes(tileSizes)
        .setLoopType(linalg::LinalgTilingLoopType::Loops);
      
      // Perform tiling
      llvm::errs() << "Tiling matmulOp directly...\n";
      FailureOr<linalg::TiledLinalgOp> tiled = 
        linalg::tileLinalgOp(rewriter, matmulOp, options);
      
      if (failed(tiled)) {
        llvm::errs() << "Tiling failed for a matmulOp!\n";
        anyFailed = true;
        continue;
      }
      
      // Replace the original op
      rewriter.replaceOp(matmulOp, tiled->tensorResults);
      llvm::errs() << "Successfully tiled a matmulOp\n";
    }
    
    if (anyFailed) {
      signalPassFailure();
    } else {
      // Print the final result
      llvm::errs() << "\nTransformed IR:\n";
      op->print(llvm::errs(), OpPrintingFlags().enableDebugInfo());
      llvm::errs() << "\n";
    }
  }
};

std::unique_ptr<Pass> createMatmulOptPass() {
  return std::make_unique<MatmulOptPass>();
}

} // namespace sample
} // namespace mlir