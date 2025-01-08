#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/IR/IRMapping.h"
#include "llvm/Support/Casting.h"
#include <iomanip>

static inline mlir::Operation *getDefiningOpByValue(mlir::Value val) {
  if (auto blockArg = val.dyn_cast<mlir::BlockArgument>()) {
    return blockArg.getOwner()->getParentOp();
  }
  return val.getDefiningOp();
}