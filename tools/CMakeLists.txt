# Get all MLIR dialect and conversion libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(sample-opt
  mlir-opt.cpp
)

llvm_update_compile_flags(sample-opt)
target_link_libraries(sample-opt
  PRIVATE
  ${dialect_libs}  # All dialect libraries
  ${conversion_libs}  # All conversion libraries
  MLIROptLib
  SamplePasses
)

# Optional check if available
if(COMMAND mlir_check_all_link_libraries)
  mlir_check_all_link_libraries(sample-opt)
endif()