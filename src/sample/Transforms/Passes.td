#ifndef _SAMPLES_TRANSFORMS_PASSES_TD_
#define _SAMPLES_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def SampleTypeCheck : Pass<"sample-type-check"> {
    let summary = "checks the type of each op"; 
    let constructor = "mlir::sample::createSampleTypeCheckPass()"; 
}

// The ID string "matmul-opt" must match your registration name
def MatmulOpt : Pass<"matmul-opt"> {
  let summary = "Optimize matrix multiplications using tiling";
  let constructor = "mlir::sample::createMatmulOptPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect"
  ];
}


def SampleConvertFunction : Pass<"sample-convert-function"> {
  let constructor = "mlir::sample::createSampleConvertFunctionPass()"; 
}

#endif //_SAMPLES_TRANSFORMS_PASSES_TD_