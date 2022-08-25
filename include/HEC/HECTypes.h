#ifndef HEC_TYPES_H
#define HEC_TYPES_H

#include "HEC/HECDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

namespace mlir
{
  namespace hec
  {
    namespace detail
    {
    } // end detail

  } // end hec
} // end mlir

//===----------------------------------------------------------------------===//
// Tablegen Type Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "HEC/HECTypes.h.inc"

namespace mlir
{
  namespace hec
  {
  } // end hec
} // end mlir

#endif // HEC_TYPES_H