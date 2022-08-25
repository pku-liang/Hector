#ifndef HEC_PASS_DETAIL_H
#define HEC_PASS_DETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir
{
  template <typename ConcreteDialect>
  void registerDialect(DialectRegistry &registry);

  namespace hec
  {
    class HECDialect;
  }

  namespace tor
  {
    class TORDialect;
  }

#define GEN_PASS_CLASSES
#include "HEC/Passes.h.inc"
}
#endif