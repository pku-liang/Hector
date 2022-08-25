module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-s128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  tor.design @stencil3d {
    %arg0 = tor.alloc : !tor.memref<2xi32, [], "r">
    %arg1 = tor.alloc : !tor.memref<16384xi32, [], "r">
    %arg2 = tor.alloc : !tor.memref<16384xi32, [], "w">

    tor.func @main()
    attributes {resource="../examples/resource_dynamatic.json", clock=6.0} {
      %c1 = constant 1 : index
      %c31 = constant 31 : index
      %c-1 = constant -1 : index
      %c15 = constant 15 : index
      %c14 = constant 14 : index
      %c0 = constant 0 : index
      %c16 = constant 16 : index
      %c4 = constant 4 : index
      %c32 = constant 32 : index
      %c5 = constant 5 : index
      %c30 = constant 30 : index
      %c992 = constant 992 : index
      %c1_i32 = constant 1 : i32
      scf.for %arg3 = %c0 to %c31 step %c1 {
        scf.for %arg4 = %c0 to %c15 step %c1 {
          %0 = shift_left %arg3, %c4 : index
          %1 = addi %arg4, %0 : index
          %2 = tor.load %arg1[%1] on (0 to 0) : !tor.memref<16384xi32, [], "r">[index]
          tor.store %2 to %arg2[%1] on (0 to 0) : (i32, !tor.memref<16384xi32, [], "w">[index])
          %3 = addi %arg3, %c992 : index
          %4 = shift_left %3, %c4 : index
          %5 = addi %arg4, %4 : index
          %6 = tor.load %arg1[%5] on (0 to 0) : !tor.memref<16384xi32, [], "r">[index]
          tor.store %6 to %arg2[%5] on (0 to 0) : (i32, !tor.memref<16384xi32, [], "w">[index])
        } {pipeline=1, II=1}
      }
      scf.for %arg3 = %c1 to %c30 step %c1 {
        // %0 = addi %arg3, %c-1 : index
	%0 = subi %arg3, %c1 : index
        %1 = index_cast %0 : index to i32
        %2 = addi %1, %c1_i32 : i32
        %3 = index_cast %2 : i32 to index
        scf.for %arg4 = %c0 to %c15 step %c1 {
          %4 = shift_left %3, %c5 : index
          %5 = shift_left %4, %c4 : index
          %6 = addi %arg4, %5 : index
          %7 = tor.load %arg1[%6] on (0 to 0) : !tor.memref<16384xi32, [], "r">[index]
          tor.store %7 to %arg2[%6] on (0 to 0) : (i32, !tor.memref<16384xi32, [], "w">[index])
          %8 = addi %4, %c31 : index
          %9 = shift_left %8, %c4 : index
          %10 = addi %arg4, %9 : index
          %11 = tor.load %arg1[%10] on (0 to 0) : !tor.memref<16384xi32, [], "r">[index]
          tor.store %11 to %arg2[%10] on (0 to 0) : (i32, !tor.memref<16384xi32, [], "w">[index])
        } {pipeline=1, II=1}
      }
      scf.for %arg3 = %c1 to %c30 step %c1 {
        // %0 = addi %arg3, %c-1 : index
	%0 = subi %arg3, %c1 : index
        %1 = index_cast %0 : index to i32
        %2 = addi %1, %c1_i32 : i32
        %3 = index_cast %2 : i32 to index
        scf.for %arg4 = %c1 to %c30 step %c1 {
          // %4 = addi %arg4, %c-1 : index
	  %4 = subi %arg4, %c1 : index
          %5 = index_cast %4 : index to i32
          %6 = addi %5, %c1_i32 : i32
          %7 = index_cast %6 : i32 to index
          %8 = shift_left %3, %c5 : index
          %9 = addi %7, %8 : index
          %10 = shift_left %9, %c4 : index
          %11 = tor.load %arg1[%10] on (0 to 0) : !tor.memref<16384xi32, [], "r">[index]
          tor.store %11 to %arg2[%10] on (0 to 0) : (i32, !tor.memref<16384xi32, [], "w">[index])
          %12 = addi %10, %c15 : index
          %13 = tor.load %arg1[%12] on (0 to 0) : !tor.memref<16384xi32, [], "r">[index]
          tor.store %13 to %arg2[%12] on (0 to 0) : (i32, !tor.memref<16384xi32, [], "w">[index])
        } {pipeline=1, II=1}
      }
      scf.for %arg3 = %c1 to %c30 step %c1 {
        %0 = subi %arg3, %c1 : index
        %1 = index_cast %0 : index to i32
        %2 = addi %1, %c1_i32 : i32
        %3 = index_cast %2 : i32 to index
        scf.for %arg4 = %c1 to %c30 step %c1 {
          %4 = subi %arg4, %c1 : index
          %5 = index_cast %4 : index to i32
          %6 = addi %5, %c1_i32 : i32
          %7 = index_cast %6 : i32 to index
          scf.for %arg5 = %c1 to %c14 step %c1 {
            %8 = subi %arg5, %c1 : index
            %9 = index_cast %8 : index to i32
            %10 = addi %9, %c1_i32 : i32
            %11 = index_cast %10 : i32 to index
            %12 = shift_left %3, %c5 : index
            %13 = addi %7, %12 : index
            %14 = shift_left %13, %c4 : index
            %15 = addi %11, %14 : index
            %16 = tor.load %arg1[%15] on (0 to 0) : !tor.memref<16384xi32, [], "r">[index]
            %17 = addi %3, %c1 : index
            %18 = shift_left %17, %c5 : index
            %19 = addi %7, %18 : index
            %20 = shift_left %19, %c4 : index
            %21 = addi %11, %20 : index
            %22 = tor.load %arg1[%21] on (0 to 0) : !tor.memref<16384xi32, [], "r">[index]
            %23 = subi %3, %c1 : index
            %24 = shift_left %23, %c5 : index
            %25 = addi %7, %24 : index
            %26 = shift_left %25, %c4 : index
            %27 = addi %11, %26 : index
            %28 = tor.load %arg1[%27] on (0 to 0) : !tor.memref<16384xi32, [], "r">[index]
            %29 = addi %22, %28 : i32
            %30 = addi %7, %c1 : index
            %31 = addi %30, %12 : index
            %32 = shift_left %31, %c4 : index
            %33 = addi %11, %32 : index
            %34 = tor.load %arg1[%33] on (0 to 0) : !tor.memref<16384xi32, [], "r">[index]
            %35 = addi %29, %34 : i32
            %36 = subi %7, %c1 : index
            %37 = addi %36, %12 : index
            %38 = shift_left %37, %c4 : index
            %39 = addi %11, %38 : index
            %40 = tor.load %arg1[%39] on (0 to 0) : !tor.memref<16384xi32, [], "r">[index]
            %41 = addi %35, %40 : i32
            %42 = addi %11, %c1 : index
            %43 = addi %42, %14 : index
            %44 = tor.load %arg1[%43] on (0 to 0) : !tor.memref<16384xi32, [], "r">[index]
            %45 = addi %41, %44 : i32
            %46 = subi %11, %c1 : index
            %47 = addi %46, %14 : index
            %48 = tor.load %arg1[%47] on (0 to 0) : !tor.memref<16384xi32, [], "r">[index]
            %49 = addi %45, %48 : i32
            %50 = tor.load %arg0[%c0] on (0 to 0) : !tor.memref<2xi32, [], "r">[index]
            %51 = muli %16, %50 : i32
            %52 = tor.load %arg0[%c1] on (0 to 0) : !tor.memref<2xi32, [], "r">[index]
            %53 = muli %49, %52 : i32
            %54 = addi %51, %53 : i32
            tor.store %54 to %arg2[%15] on (0 to 0) : (i32, !tor.memref<16384xi32, [], "w">[index])
          } {pipeline=1, II=1}
        }
      }
      tor.return
    }
  }
}
