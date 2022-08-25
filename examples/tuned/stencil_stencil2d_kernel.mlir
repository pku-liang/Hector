module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  tor.design @stencil {
    %arg0 = tor.alloc : !tor.memref<8192xi32, [], "r">
    %arg1 = tor.alloc : !tor.memref<8192xi32, [], "w">
    %arg2 = tor.alloc : !tor.memref<9xi32, [], "r">
    tor.func @main() -> ()
    attributes {resource="../examples/resource_dynamatic.json", clock=6.0} {
      %c1 = constant 1 : index
      %c11 = constant 1 : index
      %c0 = constant 0 : index
      %c126 = constant 126 : index
      %c125 = constant 125 : index
      %c62 = constant 62 : index
      %c61 = constant 61 : index
      %c3 = constant 3 : index
      %c2 = constant 2 : index
      %c64 = constant 64 : index
      %c6 = constant 6 : index
      %c0_i32 = constant 0 : i32
      scf.for %arg3 = %c0 to %c125 step %c1 {
        scf.for %arg4 = %c0 to %c61 step %c1 {
          %0 = scf.for %arg5 = %c0 to %c2 step %c1 iter_args(%arg6 = %c0_i32) -> (i32) {
            %3 = scf.for %arg8 = %c0 to %c2 step %c1 iter_args(%arg9 = %arg6) -> (i32) {
              %test = shift_left %arg5, %c11 : index
              %4 = addi %arg5, %test : index
              %5 = addi %4, %arg8 : index
              %6 = tor.load %arg2[%5] on (0 to 0) : !tor.memref<9xi32, [], "r">[index]
              %7 = addi %arg3, %arg5 : index
              %8 = shift_left %7, %c6 : index
              %9 = addi %8, %arg4 : index
              %10 = addi %9, %arg8 : index
              %11 = tor.load %arg0[%10] on (0 to 0) : !tor.memref<8192xi32, [], "r">[index]
              %12 = muli %6, %11 : i32
              %13 = addi %arg9, %12 : i32
              scf.yield %13: i32
            } {pipeline=1, II=1}
            scf.yield %3 : i32
          }
          %1 = shift_left %arg3, %c6 : index
          %2 = addi %1, %arg4 : index
          tor.store %0 to %arg1[%2] on (0 to 0) : (i32, !tor.memref<8192xi32, [], "w">[index])
        }
      }
      tor.return
    }
  }
}
