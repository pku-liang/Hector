#ifndef TOR_DIALECT_H
#define TOR_DIALECT_H

#include "TOR/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace tor {

// class TORType;

} // namespace tor
} // namespace mlir

// Pull in the dialect definition.
#include "TOR/TORDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "TOR/TOREnums.h.inc"

#endif // TOR_TORDIALECT_H
