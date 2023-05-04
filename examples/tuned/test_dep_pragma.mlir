module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  tor.design @stencil {
    %f =  tor.alloc : !tor.memref<64xi32, [], "rw">
    tor.func @main() -> ()
    attributes {resource="../examples/resource_dynamatic.json", clock=6.0} {
      %c1 = arith.constant 1 : index
      %c11 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c126 = arith.constant 126 : index
      %c125 = arith.constant 125 : index
      %c63 = arith.constant 63 : index
      %c61 = arith.constant 61 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c64 = arith.constant 64 : index
      %c6 = arith.constant 6 : index
      %c0_i32 = arith.constant 0 : i32
      scf.for %arg3 = %c0 to %c63 step %c1 {
        %idx1 = arith.subi %arg3, %c3 : index
        %idx2 = arith.subi %arg3, %c4 : index
        %a = tor.load %f[%idx1] on (0 to 0) {t=#tor.dep<(2), (3)>} : !tor.memref<64xi32, [], "rw">[index]
        %b = tor.load %f[%idx2] on (0 to 0) {d=#tor.dep<(1), (2)>}: !tor.memref<64xi32, [], "rw">[index]
        %s = arith.addi %a, %b : i32
        tor.store %s to %f[%arg3] on (0 to 0) {d=#tor.dep<(1, 2), (2, 3)>}: (i32, !tor.memref<64xi32, [], "rw">[index])
      } {pipeline=1, II=1}
      tor.return
    }
  }
}
