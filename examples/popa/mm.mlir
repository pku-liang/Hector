module @kernels {
  func.func @_kernel_C_s0_run_on_device() {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {var_name = "A.buffer"} : memref<16x16xi32>
    %alloc_0 = memref.alloc() {var_name = "B.buffer"} : memref<16x16xi32>
    %alloc_1 = memref.alloc() {var_name = "C.buffer"} : memref<16x16xi32>
    affine.for %arg0 = 0 to 16 {
      affine.for %arg1 = 0 to 16 {
        %alloc_2 = memref.alloc() : memref<1xi32>
        memref.store %c0_i32, %alloc_2[%c0] : memref<1xi32>
        affine.for %arg2 = 0 to 16 {
          %1 = affine.load %alloc_0[%arg1, %arg2] : memref<16x16xi32>
          %2 = affine.load %alloc[%arg2, %arg0] : memref<16x16xi32>
          %3 = arith.muli %2, %1 : i32
          %4 = memref.load %alloc_2[%c0] : memref<1xi32>
          %5 = arith.addi %4, %3 : i32
          memref.store %5, %alloc_2[%c0] : memref<1xi32>
        }
        %0 = memref.load %alloc_2[%c0] : memref<1xi32>
        affine.store %0, %alloc_1[%arg1, %arg0] : memref<16x16xi32>
      }
    }
    return
  }
}

