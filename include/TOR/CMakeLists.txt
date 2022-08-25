set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TOR)
add_public_tablegen_target(MLIRTORPassIncGen)

set(LLVM_TARGET_DEFINITIONS TOR.td)
mlir_tablegen(TOREnums.h.inc -gen-enum-decls)
mlir_tablegen(TOREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTOREnumsIncGen)

add_mlir_dialect(TOR tor)
# add_mlir_doc(TORDialect -gen-dialect-doc TORDialect TOR/)
# add_mlir_doc(TOR -gen-op-doc TOR TOR/)
