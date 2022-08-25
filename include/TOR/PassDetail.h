#ifndef TOR_PASS_DETAIL_H
#define TOR_PASS_DETAIL_H

#include "mlir/Pass/Pass.h"
#include "TOR/TOR.h"

namespace mlir
{
  template <typename ConcreteDialect>
  void registerDialect(DialectRegistry &registry);

  namespace tor
  {
    class TORDialect;
  }

#define GEN_PASS_CLASSES
#include "TOR/Passes.h.inc"
}
#endif //TOR_PASS_DETAIL_H