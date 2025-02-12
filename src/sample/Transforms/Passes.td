#ifndef _SAMPLES_TRANSFORMS_PASSES_TD_
#define _SAMPLES_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def SampleTypeCheck : Pass<"sample-type-check"> {
    let summary = "checks the type of each op"; 
    let constructor = "mlir::sample::createSampleTypeCheckPass()"; 
}

#endif //_SAMPLES_TRANSFORMS_PASSES_TD_