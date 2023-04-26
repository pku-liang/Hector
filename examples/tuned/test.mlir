module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  tor.design @stencil {
    tor.func @main() -> ()
    attributes {resource="../examples/resource_dynamatic.json", clock=6.0} {
      %c1 = arith.constant 1 : index
      %c11 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c126 = arith.constant 126 : index
      %c125 = arith.constant 125 : index
      %c62 = arith.constant 62 : index
      %c61 = arith.constant 61 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c6 = arith.constant 6 : index
      %c0_i32 = arith.constant 0 : i32
      scf.for %arg3 = %c0 to %c125 step %c1 {
        scf.for %arg4 = %c0 to %c61 step %c1 {
          %0 = scf.for %arg5 = %c0 to %c2 step %c1 iter_args(%arg6 = %c0_i32) -> (i32) {
            %3 = scf.for %arg8 = %c0 to %c2 step %c1 iter_args(%arg9 = %arg6) -> (i32) {
              %test = arith.shli %arg5, %c11 : index
              %4 = arith.addi %arg5, %test : index
              %5 = arith.addi %4, %arg8 : index
              %6 = arith.index_cast %5 : index to i32
              %7 = arith.addi %arg3, %arg5 : index
              %8 = arith.shli %7, %c6 : index
              %9 = arith.addi %8, %arg4 : index
              %10 = arith.addi %9, %arg8 : index
              %11 = arith.index_cast %10: index to i32
              %12 = arith.muli %6, %11 : i32
              %13 = arith.addi %arg9, %12 : i32
              scf.yield %13: i32
            } {pipeline=1, II=1}
            scf.yield %3 : i32
          }
          %1 = arith.shli %arg3, %c6 : index
          %2 = arith.addi %1, %arg4 : index
        }
      }
      tor.return
    }
  }
}
