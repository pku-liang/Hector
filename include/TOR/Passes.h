#ifndef TOR_PASSES_H
#define TOR_PASSES_H

#include "mlir/Pass/Pass.h"
#include "TOR/TOR.h"
#include <limits>

namespace mlir
{
  std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createTORSchedulePass();
  std::unique_ptr<OperationPass<mlir::ModuleOp>> createTORPipelinePartitionPass();
  std::unique_ptr<OperationPass<mlir::ModuleOp>> createTORSplitPass();
  std::unique_ptr<OperationPass<mlir::ModuleOp>> createTORCheckPass();
  std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createSCFToTORPass();
  std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createSCFDumpPass();
  std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createWidthAnalysisPass();
#define GEN_PASS_REGISTRATION
#include "TOR/Passes.h.inc"
}
#endif // TOR_PASSES_H
