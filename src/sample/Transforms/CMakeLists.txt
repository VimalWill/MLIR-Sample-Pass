set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls)

add_public_tablegen_target(MLIRSampleTransformsIncGen)

# Get all available dialect libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(SamplePasses
  SampleTypeCheckPass.cpp
  MatmulOptPass.cpp
  SampleConvertFunctionPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}
  
  DEPENDS
  MLIRSampleTransformsIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  ${dialect_libs}  # Link against all dialect libraries
)