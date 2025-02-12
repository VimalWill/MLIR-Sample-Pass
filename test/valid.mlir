module {
  func.func @test(%arg0: i8) -> i8 {
    %0 = arith.addi %arg0, %arg0 : i8
    return %0 : i8
  }
}
