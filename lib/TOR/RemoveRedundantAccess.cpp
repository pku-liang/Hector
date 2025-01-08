#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#include "TOR/AccessDependence.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#define DEBUG_TYPE "remove-access"

namespace {
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::func;
using std::string;

struct RemoveRedundantAccessPass
    : RemoveRedundantAccessBase<RemoveRedundantAccessPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    OptMode optMode = OptMode::OPT_CONSERVATIVE;
    if (mode == "none") {
      optMode = OptMode::OPT_NONE;
      return;
    } else if (mode == "aggressive") {
      optMode = OptMode::OPT_AGGRESSIVE;
    }
    moduleOp.walk([&](func::FuncOp func) {
      func.walk([&](Block *block) {
        RemoveMemeoryAccesses removeAccesses(block, optMode);
        removeAccesses.removeAccesses();
        LLVM_DEBUG(llvm::dbgs() << "Remove memeory accesses num: "
                                << removeAccesses.eraseNum << "\n");
      });
    });
  }
};
} // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createRemoveRedundantAccessPass() {
  return std::make_unique<RemoveRedundantAccessPass>();
}

} // namespace mlir
