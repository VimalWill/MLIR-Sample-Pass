#ifndef _SAMPLE_TRANSFORM_PASSES_H_
#define _SAMPLE_TRANSFORM_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::sample {

std::unique_ptr<mlir::Pass> createSampleTypeCheckPass(); 

#define GEN_PASS_REGISTRATION
#include "sample/Transforms/Passes.h.inc"

#define GEN_PASS_DECL
#include "sample/Transforms/Passes.h.inc"

} // namespace mlir::sample

#endif //_SAMPLE_TRANSFORM_PASSES_H_