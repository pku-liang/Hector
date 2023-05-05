#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "TOR/TORAttrs.h"
#include <iostream>
#include <cassert>

#define GET_ATTRDEF_CLASSES
#include "TOR/TORAttrs.cpp.inc"

namespace mlir {

  namespace tor {
    void TORDialect::registerAttributes() {
      addAttributes<
      #define GET_ATTRDEF_LIST
      #include <TOR/TORAttrs.cpp.inc>
      >();
    }

    ::mlir::LogicalResult DependenceAttr::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
        ::llvm::ArrayRef<int> signatures,
        ::llvm::ArrayRef<int> distances
    ) {
      if (signatures.size() != distances.size())
        return emitError() << "Size mismatch";
      for (auto i : distances)
        if (i < 0 && i != -1)
          return emitError() << "incorrect distance value";
      
      return success();
    }
  }
}