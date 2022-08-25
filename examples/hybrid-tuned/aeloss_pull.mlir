module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
tor.design @aeloss {
  %arg0 = tor.alloc : !tor.memref<1024xf64, [], "r">
  %arg1 = tor.alloc : !tor.memref<1024xf64, [], "r">
  %arg2 = tor.alloc : !tor.memref<1024xf64, [], "rw">
  %arg3 = tor.alloc : !tor.memref<1024xi32, [], "r">
  %c1 = constant 1 : index
  %c1024 = constant 1024 : index
  %c0 = constant 0 : index
  %false = constant false
  %c0_i32 = constant 0 : i32
  %cst = constant 4.000000e-04 : f64
  %c2_i32 = constant 2 : i32
  %cst_0 = constant 1.000000e+00 : f64
  %c1_i32 = constant 1 : i32
  %cst_1 = constant 0.000000e+00 : f64
  %cst_1_2 = constant 0.500 : f64
  %cst_1_n = constant 0.097713504 : f64
  %2 = constant 1.234e+00 : f64


  tor.func @submodule1(%arg5 : index) -> (f64, f64) 
    attributes {pipeline="func", resource="../examples/resource_dynamatic.json", clock=6.0, II=2, strategy="static"} {
    %0 = tor.load %arg0[%arg5] on (0 to 0) : !tor.memref<1024xf64, [], "r">[index]
    %1 = tor.load %arg1[%arg5] on (0 to 0) : !tor.memref<1024xf64, [], "r">[index]
    %10 = addf %0, %1 : f64
    %12 = mulf %10, %cst_1_2 : f64
    %13 = subf %0, %12 : f64
    %14 = subf %1, %12 : f64
    %15 = mulf %13, %13 : f64
    %16 = mulf %14, %14 : f64
    %17 = addf %15, %16 : f64
    %18 = mulf %17, %cst_1_n : f64
    tor.return %12, %18 : f64, f64
  }

  tor.func @main() -> (f64)
    attributes {resource="../examples/resource_dynamatic.json", clock=6.0, strategy="dynamic"} {
    %pull = scf.for %arg5 = %c0 to %c1024 step %c1 iter_args(%pull.1 = %cst_1) -> (f64) {
      %12, %18 = tor.call @submodule1(%arg5) on (0 to 0) : (index) -> (f64, f64)
      tor.store %12 to %arg2[%arg5] on (0 to 0) : (f64, !tor.memref<1024xf64, [], "rw">[index])
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
