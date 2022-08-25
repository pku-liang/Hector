module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  tor.design @gemm {
    %arg0 = tor.alloc : !tor.memref<4096xf64, [], "r">
    %arg1 = tor.alloc : !tor.memref<4096xf64, [], "r">
    %arg2 = tor.alloc : !tor.memref<4096xf64, [], "w">
    tor.func @main() -> ()
    attributes {resource="../examples/resource_dynamatic.json", clock=6.0} {
      %c1 = constant 1 : index
      %c0 = constant 0 : index
      %c64 = constant 64 : index
      %c6 = constant 6 : index
      %c63 = constant 63 : index // %c64 - 1 for loop bound
      %c0_i32 = constant 0 : i32
      scf.for %arg3 = %c0 to %c63 step %c1 {
        scf.for %arg4 = %c0 to %c63 step %c1 {
          %0 = shift_left %arg3, %c6 : index
          %1 = sitofp %c0_i32 : i32 to f64
          %2 = scf.for %arg5 = %c0 to %c63 step %c1 iter_args(%arg6 = %1) -> (f64) {
            %4 = shift_left %arg5, %c6 : index
            %5 = addi %0, %arg5 : index
            %6 = tor.load %arg0[%5] on (0 to 0) : !tor.memref<4096xf64, [], "r">[index]
            %7 = addi %4, %arg4 : index
            %8 = tor.load %arg1[%7] on (0 to 0) : !tor.memref<4096xf64, [], "r">[index]
            %9 = mulf %6, %8 : f64
            %10 = addf %arg6, %9 : f64
            scf.yield %10 : f64
          } {pipeline=1, II=1}
          %3 = addi %0, %arg4 : index
          tor.store %2 to %arg2[%3] on (0 to 0) : (f64, !tor.memref<4096xf64, [], "w">[index])
        }
      }
      tor.return
    }
  }
}
