## MLIR SAMPLE PASS 
Sample template for MLIR-based pass for type-check. it checks the functions "args" are in INT-8 data-type. 

## Build Instructions
> assuming the environment already build with MLIR 
```bash
$ export MLIR_DIR=~/llvm-project/build/lib/cmake/mlir
$ mkdir build && cd build 
$ cmake ..
$ make -j32 
```
finally, the logic can be tested as following,
```bash
./bin/sample-opt ../test/invalid.mlir --sample-typecheck
```