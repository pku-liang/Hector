#ifndef HEC_OPS_TD
#define HEC_OPS_TD

include "HECDialect.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/EnumAttr.td"

include "HECStructure.td"
include "HECExpr.td"
include "HECPipeline.td"
include "HECSTG.td"
include "HECGraph.td"

#endif // HEC_OPS_TD
