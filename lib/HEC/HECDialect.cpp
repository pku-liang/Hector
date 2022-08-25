#include "HEC/HECDialect.h"
#include "HEC/HEC.h"
#include "HEC/HECTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::hec;

void HECDialect::initialize()
{
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "HEC/HEC.cpp.inc"
      >();
}

// Provide implementations for the enums we use.
// #include "TOR/TOREnums.cpp.inc"
#include "HEC/HECEnums.cpp.inc"
