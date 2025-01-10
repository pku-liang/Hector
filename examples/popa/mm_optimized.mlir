module @kernels {
  func.func @_kernel_C_s0_run_on_device() {
    %c0_i32 = arith.constant 0 : i32
    %c15_i32 = arith.constant 15 : i32
    %alloc = memref.alloc() {partition_cyclic_array = [1 : i32], partition_dim_array = [1 : i32], partition_factor_array = [4 : i32], var_name = "A.buffer"} : memref<16x16xi32>
    %alloc_0 = memref.alloc() {partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [4 : i32], var_name = "B.buffer"} : memref<16x16xi32>
    %alloc_1 = memref.alloc() {partition_cyclic_array = [1 : i32, 1 : i32], partition_dim_array = [0 : i32, 1 : i32], partition_factor_array = [4 : i32, 4 : i32], var_name = "C.buffer"} : memref<4x4x4x4xi32>
    %alloc_2 = memref.alloc() {partition_cyclic_array = [1 : i32, 1 : i32], partition_dim_array = [0 : i32, 1 : i32], partition_factor_array = [4 : i32, 4 : i32], var_name = "Z.shreg"} : memref<4x4xi32>
    %alloc_3 = memref.alloc() {partition_cyclic_array = [1 : i32, 1 : i32], partition_dim_array = [0 : i32, 1 : i32], partition_factor_array = [4 : i32, 4 : i32], var_name = "Y.shreg"} : memref<4x4xi32>
    %alloc_4 = memref.alloc() {partition_cyclic_array = [1 : i32, 1 : i32], partition_dim_array = [0 : i32, 1 : i32], partition_factor_array = [4 : i32, 4 : i32], var_name = "X.shreg"} : memref<4x4xi32>
    affine.for %arg0 = 0 to 4 {
      affine.for %arg1 = 0 to 4 {
        affine.for %arg2 = 0 to 16 {
          %0 = arith.index_cast %arg2 : index to i32
          %1 = arith.cmpi eq, %0, %c15_i32 : i32
          %2 = arith.index_cast %arg2 : index to i32
          %3 = arith.cmpi eq, %2, %c0_i32 : i32
          affine.for %arg3 = 0 to 4 {
            %4 = affine.load %alloc[%arg2, %arg3 + %arg0 * 4] : memref<16x16xi32>
            %5 = arith.index_cast %arg3 : index to i32
            %6 = arith.cmpi eq, %5, %c0_i32 : i32
            affine.for %arg4 = 0 to 4 {
              %7 = affine.load %alloc_4[%arg4 - 1, %arg3] : memref<4x4xi32>
              %8 = arith.index_cast %arg4 : index to i32
              %9 = arith.cmpi eq, %8, %c0_i32 : i32
              %10 = arith.select %9, %4, %7 : i32
              affine.store %10, %alloc_4[%arg4, %arg3] : memref<4x4xi32>
              %11 = affine.load %alloc_3[%arg4, %arg3 - 1] : memref<4x4xi32>
              %12 = affine.load %alloc_0[%arg4 + %arg1 * 4, %arg2] : memref<16x16xi32>
              %13 = arith.select %6, %12, %11 : i32
              affine.store %13, %alloc_3[%arg4, %arg3] : memref<4x4xi32>
              %14 = affine.load %alloc_3[%arg4, %arg3] : memref<4x4xi32>
              %15 = affine.load %alloc_4[%arg4, %arg3] : memref<4x4xi32>
              %16 = arith.muli %15, %14 : i32
              %17 = affine.load %alloc_2[%arg4, %arg3] : memref<4x4xi32>
              %18 = arith.select %3, %c0_i32, %17 : i32
              %19 = arith.addi %18, %16 : i32
              affine.store %19, %alloc_2[%arg4, %arg3] : memref<4x4xi32>
            } {unroll = 0 : i32}
          } {unroll = 0 : i32}
        } {pipeline = 1 : i32}
      }
    }
    return
  }
}

