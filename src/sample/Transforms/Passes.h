#ifndef _SAMPLE_TRANSFORM_PASSES_H_
#define _SAMPLE_TRANSFORM_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::sample {

// Pass creation forward declarations
std::unique_ptr<mlir::Pass> createSampleTypeCheckPass();
std::unique_ptr<mlir::Pass> createMatmulOptPass();

// Include the auto-generated declarations
#define GEN_PASS_DECL
#include "sample/Transforms/Passes.h.inc"

// Registration function for all passes
#define GEN_PASS_REGISTRATION
#include "sample/Transforms/Passes.h.inc"

} // namespace mlir::sample

#endif //_SAMPLE_TRANSFORM_PASSES_H_