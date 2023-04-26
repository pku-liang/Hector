#ifndef TOR_OPS_H
#define TOR_OPS_H

#include "TOR/TORDialect.h"
#include "TOR/TORTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace mlir {
namespace tor {

} // namespace tor
} // namespace mlir

#define GET_OP_CLASSES
#include "TOR/TOR.h.inc"


#endif // TOR_OPS_H