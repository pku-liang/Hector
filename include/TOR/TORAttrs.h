#ifndef TOR_ATTRS_H
#define TOR_ATTRS_H

#include "mlir/IR/Attributes.h"

#include "TOR/TORDialect.h"
#include "TOR/TOR.h"
#define GET_ATTRDEF_CLASSES
#include "TOR/TORAttrs.h.inc"

namespace mlir {
  namespace tor {
    void registerAttributes();
  }
}
#endif