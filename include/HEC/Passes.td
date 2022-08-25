#ifndef HEC_PASSES
#define HEC_PASSES

include "mlir/Pass/PassBase.td"

def HECGen : Pass<"generate-hec", "mlir::ModuleOp"> {
  let summary = "Lower tor to hec";
  let constructor = "mlir::createHECGenPass()";
  let dependentDialects = ["mlir::hec::HECDialect", "mlir::tor::TORDialect"];
}

def dumpChisel : Pass<"dump-chisel", "mlir::ModuleOp"> {
    let summary = "Dump chisel program which has the same functionality with hec program";
    let constructor = "mlir::createDumpChiselPass()";
    let dependentDialects = [];
}

def dynamicSchedule : Pass<"dynamic-schedule", "mlir::ModuleOp"> {
    let summary = "Lower tor dynamic modules to hec Dialect with elastic components";
    let constructor = "mlir::createDynamicSchedulePass()";
    let dependentDialects = ["mlir::hec::HECDialect", "mlir::tor::TORDialect"];
}

#endif // HEC_PASSES
