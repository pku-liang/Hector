#ifndef HEC_PASSES_H
#define HEC_PASSES_H

#include "mlir/Pass/Pass.h"
#include <limits>

namespace mlir
{
  std::unique_ptr<OperationPass<mlir::ModuleOp>> createHECGenPass();
  std::unique_ptr<OperationPass<mlir::ModuleOp>> createDumpChiselPass();
  std::unique_ptr<OperationPass<mlir::ModuleOp>> createDynamicSchedulePass();

#define GEN_PASS_REGISTRATION
#include "HEC/Passes.h.inc"
}
#endif // HEC_PASSES_H
