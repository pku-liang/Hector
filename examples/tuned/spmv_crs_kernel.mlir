module {
  tor.design @spmv {
    %arg0 = tor.alloc : !tor.memref<1666xf64, [], "r">
    %arg1 = tor.alloc : !tor.memref<1666xi32, [], "r">
    %arg2 = tor.alloc : !tor.memref<495xi32, [], "r">
    %arg3 = tor.alloc : !tor.memref<494xf64, [], "r">
    %arg4 = tor.alloc : !tor.memref<494xf64, [], "w">
    tor.func @main () -> ()
    attributes {resource="../examples/resource_dynamatic.json", clock=6.0} {
      %c1 = constant 1 : index
      %c0 = constant 0 : index
      %c494 = constant 494 : index
      %c493 = constant 493 : index
      %c0_i32 = constant 0 : i32
      scf.for %arg5 = %c0 to %c493 step %c1 {
        %0 = sitofp %c0_i32 : i32 to f64
        %1 = tor.load %arg2[%arg5] on (0 to 0) : !tor.memref<495xi32, [], "r">[index]
        %2 = addi %arg5, %c1 : index
        %3 = tor.load %arg2[%2] on (0 to 0) : !tor.memref<495xi32, [], "r">[index]
        %4 = index_cast %3 : i32 to index
        %44 = subi %4, %c1 : index
        %5 = index_cast %1 : i32 to index
        %6 = scf.for %arg6 = %5 to %44 step %c1 iter_args(%arg7 = %0) -> (f64) {
          %7 = subi %arg6, %5 : index
          %8 = index_cast %7 : index to i32
          %9 = addi %8, %1 : i32
          %10 = index_cast %9 : i32 to index
          %11 = tor.load %arg0[%10] on (0 to 0) : !tor.memref<1666xf64, [], "r">[index]
          %12 = tor.load %arg1[%10] on (0 to 0) : !tor.memref<1666xi32, [], "r">[index]
          %13 = index_cast %12 : i32 to index
          %14 = tor.load %arg3[%13] on (0 to 0) : !tor.memref<494xf64, [], "r">[index]
          %15 = mulf %11, %14 : f64
          %16 = addf %arg7, %15 : f64
          scf.yield %16: f64
        } {pipeline=1, II=1}
        tor.store %6 to %arg4[%arg5] on (0 to 0) : (f64, !tor.memref<494xf64, [], "w">[index])
      }
      tor.return
    }
  }
}
