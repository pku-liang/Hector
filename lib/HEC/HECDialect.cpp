#include "HEC/HECDialect.h"
#include "HEC/HEC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
// #include "HEC/HECTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::hec;

#include "HEC/HECDialect.cpp.inc"

void HECDialect::initialize() {
  // registerTypes();
    addOperations<
  #define GET_OP_LIST
  #include "HEC/HEC.cpp.inc"
        >();
}

mlir::Operation *HECDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
  return builder.create<arith::ConstantOp>(loc, type, value);
}


// Provide implementations for the enums we use.
// #include "TOR/TOREnums.cpp.inc"
// #include "HEC/HECEnums.cpp.inc"
