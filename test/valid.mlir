// RUN: sample-opt --matmul-opt %s | FileCheck %s

func.func @matmul_test(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>) 
                     outs(%A : tensor<64x64xf32>) -> tensor<64x64xf32>
  // CHECK: scf.for
  // CHECK: linalg.matmul
  return %0 : tensor<64x64xf32>
}