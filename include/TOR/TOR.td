#ifndef TOR_OPS_TD
#define TOR_OPS_TD

include "TORDialect.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/EnumAttr.td"

include "TORTypes.td"

include "TORStmt.td"

include "TORExpr.td"

include "TORAttrs.td"

#endif // TOR_OPS_TD
