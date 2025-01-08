#ifndef UNROLL_UTILS_H
#define UNROLL_UTILS_H

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Support/MathExtras.h"

namespace {
using namespace mlir;
/// Computes the cleanup loop lower bound of the loop being unrolled with
/// the specified unroll factor; this bound will also be upper bound of the main
/// part of the unrolled loop. Computes the bound as an AffineMap with its
/// operands or a null map when the trip count can't be expressed as an affine
/// expression.
static void
getCleanupLoopLowerBound(AffineForOp forOp, unsigned unrollFactor,
                         AffineMap &cleanupLbMap,
                         SmallVectorImpl<Value> &cleanupLbOperands) {
  AffineMap tripCountMap;
  SmallVector<Value, 4> tripCountOperands;
  getTripCountMapAndOperands(forOp, &tripCountMap, &tripCountOperands);
  // Trip count can't be computed.
  if (!tripCountMap) {
    cleanupLbMap = AffineMap();
    return;
  }

  OpBuilder b(forOp);
  auto lbMap = forOp.getLowerBoundMap();
  auto lb = b.create<AffineApplyOp>(forOp.getLoc(), lbMap,
                                    forOp.getLowerBoundOperands());

  // For each upper bound expr, get the range.
  // Eg: affine.for %i = lb to min (ub1, ub2),
  // where tripCountExprs yield (tr1, tr2), we create affine.apply's:
  // lb + tr1 - tr1 % ufactor, lb + tr2 - tr2 % ufactor; the results of all
  // these affine.apply's make up the cleanup loop lower bound.
  SmallVector<AffineExpr, 4> bumpExprs(tripCountMap.getNumResults());
  SmallVector<Value, 4> bumpValues(tripCountMap.getNumResults());
  int64_t step = forOp.getStep();
  for (unsigned i = 0, e = tripCountMap.getNumResults(); i < e; i++) {
    auto tripCountExpr = tripCountMap.getResult(i);
    bumpExprs[i] = (tripCountExpr - tripCountExpr % unrollFactor) * step;
    auto bumpMap = AffineMap::get(tripCountMap.getNumDims(),
                                  tripCountMap.getNumSymbols(), bumpExprs[i]);
    bumpValues[i] =
        b.create<AffineApplyOp>(forOp.getLoc(), bumpMap, tripCountOperands);
  }

  SmallVector<AffineExpr, 4> newUbExprs(tripCountMap.getNumResults());
  for (unsigned i = 0, e = bumpExprs.size(); i < e; i++)
    newUbExprs[i] = b.getAffineDimExpr(0) + b.getAffineDimExpr(i + 1);

  cleanupLbOperands.clear();
  cleanupLbOperands.push_back(lb);
  cleanupLbOperands.append(bumpValues.begin(), bumpValues.end());
  cleanupLbMap = AffineMap::get(1 + tripCountMap.getNumResults(), 0, newUbExprs,
                                b.getContext());
  // Simplify the cleanupLbMap + cleanupLbOperands.
  fullyComposeAffineMapAndOperands(&cleanupLbMap, &cleanupLbOperands);
  cleanupLbMap = simplifyAffineMap(cleanupLbMap);
  canonicalizeMapAndOperands(&cleanupLbMap, &cleanupLbOperands);
  // Remove any affine.apply's that became dead from the simplification above.
  for (auto v : bumpValues)
    if (v.use_empty())
      v.getDefiningOp()->erase();

  if (lb.use_empty())
    lb.erase();
}

/// Generates unrolled copies of AffineForOp 'loopBodyBlock', with associated
/// 'forOpIV' by 'unrollFactor', calling 'ivRemapFn' to remap 'forOpIV' for each
/// unrolled body. If specified, annotates the Ops in each unrolled iteration
/// using annotateFn.
static void generateUnrolledLoop(
    Block *loopBodyBlock, Value forOpIV, uint64_t unrollFactor,
    function_ref<Value(unsigned, Value, OpBuilder)> ivRemapFn,
    function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn,
    ValueRange iterArgs, ValueRange yieldedValues) {
  // Builder to insert unrolled bodies just before the terminator of the body of
  // 'forOp'.
  auto builder = OpBuilder::atBlockTerminator(loopBodyBlock);

  if (!annotateFn)
    annotateFn = [](unsigned, Operation *, OpBuilder) {};

  // Keep a pointer to the last non-terminator operation in the original block
  // so that we know what to clone (since we are doing this in-place).
  Block::iterator srcBlockEnd = std::prev(loopBodyBlock->end(), 2);

  // Unroll the contents of 'forOp' (append unrollFactor - 1 additional copies).
  SmallVector<Value, 4> lastYielded(yieldedValues);

  for (unsigned i = 1; i < unrollFactor; i++) {
    BlockAndValueMapping operandMap;

    // Prepare operand map.
    operandMap.map(iterArgs, lastYielded);

    // If the induction variable is used, create a remapping to the value for
    // this unrolled instance.
    if (!forOpIV.use_empty()) {
      Value ivUnroll = ivRemapFn(i, forOpIV, builder);
      operandMap.map(forOpIV, ivUnroll);
    }

    // Clone the original body of 'forOp'.
    for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++) {
      Operation *clonedOp = builder.clone(*it, operandMap);
      annotateFn(i, clonedOp, builder);
    }

    // Update yielded values. If the yielded value is defined outside the
    // `loopBodyBlock` or if it is a BlockArgument then it won't be cloned, thus
    // the `lastYielded` value remains unchanged. Else, update the `lastYielded`
    // value with the clone corresponding to the yielded value.
    for (unsigned i = 0, e = lastYielded.size(); i < e; i++) {
      Operation *defOp = yieldedValues[i].getDefiningOp();
      if (defOp && defOp->getBlock() == loopBodyBlock)
        lastYielded[i] = operandMap.lookup(yieldedValues[i]);
    }
  }

  // Make sure we annotate the Ops in the original body. We do this last so that
  // any annotations are not copied into the cloned Ops above.
  for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++)
    annotateFn(0, &*it, builder);

  // Update operands of the yield statement.
  loopBodyBlock->getTerminator()->setOperands(lastYielded);
}

/// Helper to generate cleanup loop for unroll or unroll-and-jam when the trip
/// count is not a multiple of `unrollFactor`.
static LogicalResult hlsGenerateCleanupLoopForUnroll(AffineForOp forOp,
                                                     uint64_t unrollFactor) {
  // Insert the cleanup loop right after 'forOp'.
  OpBuilder builder(forOp->getBlock(), std::next(Block::iterator(forOp)));
  auto cleanupForOp = cast<AffineForOp>(builder.clone(*forOp));

  // Update uses of `forOp` results. `cleanupForOp` should use `forOp` result
  // and produce results for the original users of `forOp` results.
  auto results = forOp.getResults();
  auto cleanupResults = cleanupForOp.getResults();
  auto cleanupIterOperands = cleanupForOp.getIterOperands();

  for (auto e : llvm::zip(results, cleanupResults, cleanupIterOperands)) {
    std::get<0>(e).replaceAllUsesWith(std::get<1>(e));
    cleanupForOp->replaceUsesOfWith(std::get<2>(e), std::get<0>(e));
  }

  AffineMap cleanupMap;
  SmallVector<Value, 4> cleanupOperands;
  getCleanupLoopLowerBound(forOp, unrollFactor, cleanupMap, cleanupOperands);
  if (!cleanupMap)
    return failure();

  cleanupForOp.setLowerBound(cleanupOperands, cleanupMap);
  // Promote the loop body up if this has turned into a single iteration loop.
  // Don't need, loopUnroll can get the result
  // (void)promoteIfSingleIteration(cleanupForOp);
  (void)loopUnrollFull(cleanupForOp);
  // Adjust upper bound of the original loop; this is the same as the lower
  // bound of the cleanup loop.
  forOp.setUpperBound(cleanupOperands, cleanupMap);
  return success();
}

/// Unrolls this loop by the specified factor. Returns success if the loop
/// is successfully unrolled.
LogicalResult hlsLoopUnrollByFactor(AffineForOp forOp, uint64_t unrollFactor) {
  assert(unrollFactor > 0 && "unroll factor should be positive");

  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(forOp);
  if (unrollFactor == 1) {
    if (mayBeConstantTripCount && *mayBeConstantTripCount == 1 &&
        failed(promoteIfSingleIteration(forOp)))
      return failure();
    return success();
  }

  // Nothing in the loop body other than the terminator.
  if (llvm::hasSingleElement(forOp.getBody()->getOperations()))
    return success();

  // If the trip count is lower than the unroll factor, no unrolled body.
  if (mayBeConstantTripCount && *mayBeConstantTripCount < unrollFactor) {
    return failure();
  }

  // Generate the cleanup loop if trip count isn't a multiple of unrollFactor.
  if (getLargestDivisorOfTripCount(forOp) % unrollFactor != 0) {
    // Loops where the lower bound is a max expression or the upper bound is
    // a min expression and the trip count doesn't divide the unroll factor
    // can't be unrolled since the lower bound of the cleanup loop in such cases
    // cannot be expressed as an affine function or a max over affine functions.
    if (forOp.getLowerBoundMap().getNumResults() != 1 ||
        forOp.getUpperBoundMap().getNumResults() != 1)
      return failure();
    if (failed(hlsGenerateCleanupLoopForUnroll(forOp, unrollFactor)))
      assert(false && "cleanup loop lower bound map for single result lower "
                      "and upper bound maps can always be determined");
  }

  ValueRange iterArgs(forOp.getRegionIterArgs());
  auto yieldedValues = forOp.getBody()->getTerminator()->getOperands();

  // Scale the step of loop being unrolled by unroll factor.
  int64_t step = forOp.getStep();
  forOp.setStep(step * unrollFactor);
  generateUnrolledLoop(
      forOp.getBody(), forOp.getInductionVar(), unrollFactor,
      [&](unsigned i, Value iv, OpBuilder b) {
        // iv' = iv + i * step
        auto d0 = b.getAffineDimExpr(0);
        auto bumpMap = AffineMap::get(1, 0, d0 + i * step);
        return b.create<AffineApplyOp>(forOp.getLoc(), bumpMap, iv);
      },
      /*annotateFn=*/nullptr,
      /*iterArgs=*/iterArgs, /*yieldedValues=*/yieldedValues);

  // Promote the loop body up if this has turned into a single iteration loop.
  (void)promoteIfSingleIteration(forOp);
  return success();
}

} // namespace
#endif
