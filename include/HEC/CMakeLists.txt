set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name HEC)
add_public_tablegen_target(MLIRHECPassIncGen)

set(LLVM_TARGET_DEFINITIONS HEC.td)
mlir_tablegen(HECEnums.h.inc -gen-enum-decls)
mlir_tablegen(HECEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHECEnumsIncGen)

add_mlir_dialect(HEC hec)
# add_mlir_doc(HECDialect -gen-dialect-doc HECDialect HEC/)
# add_mlir_doc(HEC -gen-op-doc HEC HEC/)
