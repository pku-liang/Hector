#ifndef HEC_PASSES_H
#define HEC_PASSES_H

#include "mlir/Pass/Pass.h"
#include "HEC/HEC.h"
#include <limits>

namespace mlir {
    std::unique_ptr<OperationPass<mlir::ModuleOp>> createHECGenPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createDumpChiselPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createDynamicSchedulePass();

    std::unique_ptr<OperationPass<hec::DesignOp>> createHECDumpPass();

#define GEN_PASS_REGISTRATION

#include "HEC/Passes.h.inc"

}
#endif // HEC_PASSES_H
