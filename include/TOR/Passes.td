#ifndef TOR_PASSES
#define TOR_PASSES

include "mlir/Pass/PassBase.td"

def TORSchedule : Pass<"schedule-tor", "mlir::tor::DesignOp"> {
  let summary = "Lower std+scf+affine to tor, schedule a  time graph";
  let constructor = "mlir::createTORSchedulePass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def TORPipelinePartition : Pass<"pipeline-partition", "mlir::ModuleOp"> {
  let summary = "Partition on time graph in a pipelined structure";
  let constructor = "mlir::createTORPipelinePartitionPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def TORSplit : Pass<"split-schedule", "mlir::ModuleOp"> {
  let summary = "Split mixed scheduling modules";
  let constructor = "mlir::createTORSplitPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def TORCheck : Pass<"check-schedule", "mlir::ModuleOp"> {
  let summary = "Check scheduling modules";
  let constructor = "mlir::createTORCheckPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def SCFToTOR : Pass<"scf-to-tor", "mlir::tor::DesignOp"> {
  let summary = "Convert SCF to unscheduled TOR";
  let constructor = "mlir::createSCFToTORPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def SCFDump : Pass<"dump-scf", "mlir::tor::DesignOp"> {
  let summary = "Dump SCF information";
  let constructor = "mlir::createSCFDumpPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def TORDump : Pass<"dump-tor", "mlir::tor::DesignOp"> {
  let summary = "Dump TOR information";
  let constructor = "mlir::createTORDumpPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def WidthAnalysis : Pass<"width-analysis", "mlir::tor::DesignOp"> {
  let summary = "bitwidth analysis and reduction";
  let constructor = "mlir::createWidthAnalysisPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

#endif // TOR_PASSES
