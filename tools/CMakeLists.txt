get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(sample-opt
  mlir-opt.cpp
)

llvm_update_compile_flags(sample-opt)

target_link_libraries(sample-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIROptLib
  MLIRTransforms
  MLIRParser
  SamplePasses
)