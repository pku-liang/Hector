#ifndef HEC_OPS_H
#define HEC_OPS_H

#include "HEC/HECDialect.h"
#include "HEC/HECTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir
{
  namespace hec
  {
    enum PortDirection
    {
      INPUT = 0,
      OUTPUT = 1,
      INOUT = 2
    };

    struct ComponentPortInfo
    {
      StringAttr name;
      Type type;
      PortDirection direction;
      ComponentPortInfo(StringAttr name, Type type, PortDirection direction)
          : name(name), type(type), direction(direction) {}
    };

    SmallVector<ComponentPortInfo> getComponentPortInfo(Operation *op);

  } // namespace hec
} // namespace mlir

#define GET_OP_CLASSES
#include "HEC/HEC.h.inc"

#endif // HEC_OPS_H