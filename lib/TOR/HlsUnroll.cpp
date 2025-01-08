#include "TOR/PassDetail.h"
#include "TOR/Passes.h"
#include "TOR/UnrollUtils.h"
#include "TOR/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hls-unroll"

namespace {
using namespace mlir;

int64_t getSCFConstantTripCount(scf::ForOp forOp) {
  auto lbCstOp = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ubCstOp = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  assert((lbCstOp && ubCstOp && stepCstOp) &&
         "Some for loop can't unroll full, please check pragma info");
  return mlir::ceilDiv(ubCstOp.value() - lbCstOp.value(), stepCstOp.value());
}

unsigned getAttrInterger(Operation *forOp, llvm::Twine type) {
  return forOp->getAttr(type.str())
      .cast<IntegerAttr>()
      .getValue()
      .getSExtValue();
}

unsigned getUnrollAttrInterger(Operation *forOp) {
  return getAttrInterger(forOp, "unroll");
}

void forOpFailLog(Operation *forOp, std::string type) {
  // llvm::errs() << "warning: line " << getLineAttrInterger(forOp, type) << " "
  //              << type << " pragma is failed";
}

void unrollForFailLog(Operation *forOp) {
  forOpFailLog(forOp, "unroll");
  unsigned factor = getUnrollAttrInterger(forOp);
  if (factor) {
    llvm::errs() << " with factor = " << factor;
  }
}

bool checkLoopTripCount(scf::ForOp forOp) {
  auto lbCstOp = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ubCstOp = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!(lbCstOp && ubCstOp && stepCstOp)) {
    unrollForFailLog(forOp);
    llvm::errs() << ", because loop range has variable\n";
    return false;
  }
  return true;
}

void checkLogicalResultAndLog(LogicalResult result, Operation *forOp,
                              int newValue, ModuleOp moduleOp) {
  auto builder = OpBuilder(moduleOp.getContext());
  // int lineNumber = getLineAttrInterger(forOp, "unroll");
  // if (!moduleOp->getAttr("hls.pragma_line_" + llvm::Twine(lineNumber).str())) {
  //   return;
  // }
  // auto pragmaStructureAttr = llvm::to_vector<4>(
  //     moduleOp->getAttr("hls.pragma_line_" + llvm::Twine(lineNumber).str())
  //         .dyn_cast<ArrayAttr>());
  if (failed(result)) {
    unrollForFailLog(forOp);
    llvm::errs() << "\n";
    // setPragmaStructureAttrInvalid(pragmaStructureAttr, builder.getContext());
  } else {
    // setPragmaStructureAttrValid(pragmaStructureAttr, builder.getContext());
    // setPragmaStructureAttrNewValue(pragmaStructureAttr, builder.getContext(),
    //                                newValue);
  }
  // moduleOp->setAttr(
  //     "hls.pragma_line_" + llvm::Twine(lineNumber).str(),
  //     mlir::ArrayAttr::get(builder.getContext(), pragmaStructureAttr));
}

template <typename T>
void filterTypeAttrVector(ModuleOp moduleOp, SmallVector<T, 4> &loops,
                          std::string type) {
  moduleOp.walk([&](T forOp) {
    if (forOp->hasAttr(type)) {
      loops.push_back(forOp);
    }
  });
}

unsigned getPipelineAttrInterger(Operation *forOp) {
  return getAttrInterger(forOp, "pipeline");
}

void piplineForFailLog(Operation *forOp) {
  forOpFailLog(forOp, "pipeline");
  unsigned II = getPipelineAttrInterger(forOp);
  llvm::errs() << " with II = " << II;
  llvm::errs() << ", because inner code include loop\n";
}

template <typename S, typename T> void unrollInnerLoop(S op) {
  SmallVector<T, 4> innerLoops;
  op.walk([&](T innerLoop) {
    if (op != innerLoop) {
      innerLoops.push_back(innerLoop);
    }
  });
  for (auto innerLoop : innerLoops) {
    if constexpr (std::is_same_v<AffineForOp, T>) {
      (void)loopUnrollFull(innerLoop);
    } else if constexpr (std::is_same_v<scf::ForOp, T>) {
      if (unsigned factor = getSCFConstantTripCount(innerLoop)) {
        (void)loopUnrollByFactor(innerLoop, factor);
      }
    }
  }
}

template <typename S, typename T>
void unrollPipelineInnerLoop(ModuleOp moduleOp, SmallVector<S, 4> &ops) {
  ops.clear();
  filterTypeAttrVector(moduleOp, ops, "pipeline");
  for (auto op : ops) {
    unrollInnerLoop<S, T>(op);
  }

  auto hasInnerLoop = [&](S op) -> bool {
    WalkResult result = op->walk([&](T innerForOp) {
      if (op != innerForOp) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return true;
    }
    return false;
  };

  // remove illegal pipeline Attr
  for (auto op : ops) {
    if (hasInnerLoop(op)) {
      // setPragmaStructureAttrStatusByOp(op, "pipeline", false);
      piplineForFailLog(op);
      op->removeAttr("pipeline");
      op->removeAttr("II");
    }
  }
}

struct HlsUnrollPass : HlsUnrollBase<HlsUnrollPass> {
  void runOnOperation() override {
    auto moudle = getOperation();
    SmallVector<AffineForOp, 4> loops;
    filterTypeAttrVector(getOperation(), loops, "unroll");
    for (auto forOp : loops) {
      LogicalResult result(failure());
      unsigned factor = getUnrollAttrInterger(forOp);
      if (factor) {
        result = hlsLoopUnrollByFactor(forOp, factor);
      } else {
        Optional<uint64_t> mayBeConstantTripCount =
            getConstantTripCount(forOp);
        if (mayBeConstantTripCount.has_value()) {
          factor = *mayBeConstantTripCount;
        }
        result = loopUnrollFull(forOp);
      }
      checkLogicalResultAndLog(result, forOp, factor, moudle);
    }

    SmallVector<scf::ForOp, 4> scfLoops;
    filterTypeAttrVector(getOperation(), scfLoops, "unroll");
    for (auto forOp : scfLoops) {
      unsigned factor = getUnrollAttrInterger(forOp);
      if (!factor) {
        if (checkLoopTripCount(forOp)) {
          factor = getSCFConstantTripCount(forOp);
        }
      }
      LogicalResult result = loopUnrollByFactor(forOp, factor);
      checkLogicalResultAndLog(result, forOp, factor, moudle);
    }

    SmallVector<func::FuncOp, 4> funcOps;

    unrollPipelineInnerLoop<AffineForOp, AffineForOp>(getOperation(), loops);
    unrollPipelineInnerLoop<scf::ForOp, scf::ForOp>(getOperation(), scfLoops);
    unrollPipelineInnerLoop<func::FuncOp, AffineForOp>(getOperation(), funcOps);
    unrollPipelineInnerLoop<func::FuncOp, scf::ForOp>(getOperation(), funcOps);
  }
};
} // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createHlsUnrollPass() {
  return std::make_unique<HlsUnrollPass>();
}

} // namespace mlir
