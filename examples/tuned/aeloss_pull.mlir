module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
tor.design @aeloss_pull {
  %arg0 = tor.alloc : !tor.memref<1024xf64, [], "r">
  %arg1 = tor.alloc : !tor.memref<1024xf64, [], "r">
  %arg2 = tor.alloc : !tor.memref<1024xf64, [], "w">
  %arg3 = tor.alloc : !tor.memref<1024xi32, [], "r">
  tor.func @main() -> (f64) attributes {resource="../examples/resource_dynamatic.json", clock=6.0} {
    %c1 = constant 1 : index
    %c1024 = constant 1024 : index
    %c0 = constant 0 : index
    %false = constant false
    %c0_i32 = constant 0 : i32
    %cst = constant 4.000000e-04 : f64
    //%c2_i32 = constant 2 : i32
    %cst_1_2 = constant 0.500 : f64
    %cst_1_n = constant 0.097713504 : f64
    %cst_0 = constant 1.000000e+00 : f64
    %c1_i32 = constant 1 : i32
    %cst_1 = constant 0.000000e+00 : f64
    %1 = constant 10.234e+00 : f64
    %2 = subf %1, %cst_0 : f64
    %3 = mulf %1, %2 : f64
    %pull = scf.for %arg5 = %c0 to %c1024 step %c1 iter_args(%pull.1 = %cst_1) -> (f64) {
      %8 = tor.load %arg0[%arg5] on (0 to 0) : !tor.memref<1024xf64, [], "r">[index]
      %9 = tor.load %arg1[%arg5] on (0 to 0) : !tor.memref<1024xf64, [], "r">[index]
      %10 = addf %8, %9 : f64
//      %11 = sitofp %c2_i32 : i32 to f64
 //     %12 = divf %10, %11 : f64
 	%12 = mulf %10, %cst_1_2 : f64
      tor.store %12 to %arg2[%arg5] on (0 to 0) : (f64, !tor.memref<1024xf64, [], "w">[index])
      %13 = subf %8, %12 : f64
      %14 = subf %9, %12 : f64
      %15 = mulf %13, %13 : f64
      %16 = mulf %14, %14 : f64
      %17 = addf %15, %16 : f64
//      %18 = divf %17, %1 : f64
	%18 = mulf %17, %cst_1_n : f64
      %19 = tor.load %arg3[%arg5] on (0 to 0) : !tor.memref<1024xi32, [], "r">[index]
      %20 = trunci %19 : i32 to i1
      %pull.2 = scf.if %20 -> (f64) {
        %23 = addf %pull.1, %18 : f64
        scf.yield %23 : f64
      } else {
      	scf.yield %pull.1 : f64
      }
      scf.yield %pull.2 : f64
    } {pipeline=1, II=1}
    
    tor.return %pull : f64
  }
  }
}
