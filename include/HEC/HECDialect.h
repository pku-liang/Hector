#ifndef HEC_DIALECT_H
#define HEC_DIALECT_H

#include "HEC/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir
{
  namespace hec
  {

  } // namespace hec
} // namespace mlir

// Pull in the dialect definition.
#include "HEC/HECDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "HEC/HECEnums.h.inc"

#endif // HEC_DIALECT_H
