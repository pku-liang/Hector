module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
tor.design @aeloss {
  %arg0 = tor.alloc : !tor.memref<1024xf64, [], "r">
  %arg1 = tor.alloc : !tor.memref<1024xf64, [], "r">
  %arg2 = tor.alloc : !tor.memref<1024xf64, [], "r">
  %arg3 = tor.alloc : !tor.memref<1024xi32, [], "r">
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 4.000000e-04 : f64
  %c2_i32 = arith.constant 2 : i32
  %cst_0 = arith.constant 1.000000e+00 : f64
  %c1_i32 = arith.constant 1 : i32
  %cst_1 = arith.constant 0.000000e+00 : f64
  %n1 = arith.constant 1.234e+00 : f64
  %3 = arith.constant 321.0000 : f64
  
  tor.func @submodule2(%0 : f64, %1 : f64) -> (f64) 
    attributes {pipeline="func", II=2, resource="../examples/resource_dynamatic.json", clock=6.0, strategy="static"} {
    %13 = arith.subf %0, %1 : f64
    %14 = arith.sitofp %c0_i32 : i32 to f64
    %15 = arith.cmpf ugt, %13, %cst_1 : f64
    %haha0 = arith.subf %cst_0, %13 : f64
    %haha1 = arith.addf %cst_0, %13 : f64
    %16 = arith.select %15, %haha0, %haha1 : f64
    %17 = arith.cmpf ugt, %16, %14 : f64
    %18 = arith.select %17, %16, %14 : f64
    tor.return %18 : f64
  }

  tor.func @submodule3(%0 : f64, %1 : f64) -> (f64)
    attributes {pipeline="func", II=2, resource="../examples/resource_dynamatic.json", clock=6.0, strategy="static"} {
    %22 = arith.sitofp %c1_i32 : i32 to f64
    %23 = arith.addf %n1, %cst : f64
    %24 = arith.divf %22, %23 : f64
    %25 = arith.subf %0, %24 : f64
    %26 = arith.addf %3, %cst : f64
    %27 = arith.divf %25, %26 : f64
    %29 = arith.addf %1, %27 : f64
    tor.return %29 : f64
  }

  tor.func @main() -> (f64) 
    attributes {resource="../examples/resource_dynamatic.json", clock=6.0, strategy="dynamic"} {
      %push = scf.for %arg5 = %c0 to %c1024 step %c1 iter_args(%push.1 = %cst_1) -> (f64) {
      %7 = arith.index_cast %arg5 : index to i32
      %8 = tor.load %arg3[%arg5] on (0 to 0) : !tor.memref<1024xi32, [], "r">[index]
      %9 = arith.trunci %8 : i32 to i1
      %push.2 = scf.if %9 -> (f64) {
        %push.4 = scf.for %arg6 = %c0 to %c1024 step %c1 iter_args(%push.3 = %push.1) -> (f64) {
          %10 = arith.index_cast %arg6 : index to i32
          %11 = tor.load %arg2[%arg5] on (0 to 0) : !tor.memref<1024xf64, [], "r">[index]
          %12 = tor.load %arg2[%arg6] on (0 to 0) : !tor.memref<1024xf64, [], "r">[index]

          %18 = tor.call @submodule2(%11, %12) on (0 to 0) : (f64, f64) -> (f64)

          %19 = tor.load %arg3[%arg6] on (0 to 0) : !tor.memref<1024xi32, [], "r">[index]
          %20 = arith.trunci %19 : i32 to i1
          %200 = arith.cmpi ne, %7, %10 : i32
          %21 = arith.andi %20, %200 : i1

          %push.5 = scf.if %21 -> (f64) {
            %29 = tor.call @submodule3(%18, %push.3) on (0 to 0): (f64, f64) -> (f64)
            scf.yield %29 : f64
          } else {
            scf.yield %push.3 : f64
          }
          scf.yield %push.5 : f64
        } {pipeline=1, II=1}
        scf.yield %push.4 : f64
      } else {
      	scf.yield %push.1 : f64
      }
      scf.yield %push.2 : f64
    }
    tor.return %push : f64
  }
}
}
