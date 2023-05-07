#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"


#include "TOR/TORDialect.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#include <set>
#include <map>
#include <iostream>

#define DEBUG_TYPE "array-partition"

namespace {
    using namespace mlir;
    using namespace mlir::func;

    std::map<detail::ValueImpl *, int> arg_num;

    int get_idx(Value idx) {
        auto impl = idx.getImpl();
        if (arg_num.find(impl) == arg_num.end()) {
            arg_num[impl] = arg_num.size();
        }
        return arg_num[impl];
    }

    AffineExpr get_bank_expr(Value idx, PatternRewriter &rewriter) {
        if (isa<BlockArgument>(idx)) {
            return getAffineDimExpr(get_idx(idx), rewriter.getContext());
        } else if (auto apply = dyn_cast<AffineApplyOp>(idx.getDefiningOp())) {
            auto map = apply.getAffineMap();
            return map.getResult(0);
        } else if (auto constant = dyn_cast<arith::ConstantIntOp>(idx.getDefiningOp())) {
            return getAffineConstantExpr(constant.value(), rewriter.getContext());
        } else if (auto constant = dyn_cast<arith::ConstantIndexOp>(idx.getDefiningOp())) {
            return getAffineConstantExpr(constant.value(), rewriter.getContext());
        } else if (auto index_cast = dyn_cast<arith::IndexCastOp>(idx.getDefiningOp())) {
            return get_bank_expr(index_cast.getOperand(), rewriter);
        } else if (auto add = dyn_cast<arith::AddIOp>(idx.getDefiningOp())) {
            auto left = get_bank_expr(add.getOperand(0), rewriter);
            auto right = get_bank_expr(add.getOperand(1), rewriter);
            if (!left || !right) {
                return AffineExpr();
            }
            return  left + right; 
        } else if (auto mul = dyn_cast<arith::MulIOp>(idx.getDefiningOp())) {
            auto left = get_bank_expr(mul.getOperand(0), rewriter);
            auto right = get_bank_expr(mul.getOperand(1), rewriter);
            if (!left || !right) {
                return AffineExpr();
            }
            return  left * right; 
        } else if (auto sub = dyn_cast<arith::SubIOp>(idx.getDefiningOp())) {
            auto left = get_bank_expr(sub.getOperand(0), rewriter);
            auto right = get_bank_expr(sub.getOperand(1), rewriter);
            if (!left || !right) {
                return AffineExpr();
            }
            return  left - right; 
        } else {
            return AffineExpr();
        }
    }

    int get_bank(Value idx, PatternRewriter &rewriter, int factor) {
        auto expr = get_bank_expr(idx, rewriter);
        if (!expr) {
            return -1;
        }
        expr = expr % factor;
        auto map = AffineMap::get(arg_num.size(), 0, expr);
        if (map.isConstant()) {
            return map.getConstantResults()[0];
        }
        return -1;
    }

    AffineApplyOp get_new_address(Value idx, PatternRewriter &rewriter, int factor) {
        auto expr = getAffineDimExpr(0, rewriter.getContext()).ceilDiv(factor);
        auto map = AffineMap::get(1, 0, expr);
        SmallVector<Value> apply;
        apply.push_back(idx);
        return rewriter.create<AffineApplyOp>(idx.getLoc(), map, apply);
    }

    struct DesignOpPattern : OpRewritePattern<FuncOp> {
        DesignOpPattern(MLIRContext *ctx, int factor)
                : OpRewritePattern<FuncOp>(ctx), factor(factor) {}

        LogicalResult matchAndRewrite(FuncOp op,
                                      PatternRewriter &rewriter) const override {
            if (op->hasAttr("array-partition"))
                return failure();
            op->setAttr("array-partition", IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
            
            for (auto &arg : op.getArguments()) {
                if (auto memref = dyn_cast<MemRefType>(arg.getType())) {
                    for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
                        for (auto &use : arg.getUses()) {
                            auto sop = use.getOwner();
                            if (auto load = dyn_cast<memref::LoadOp>(sop)) {
                                get_bank_expr(load.getIndices()[rank], rewriter);
                            } else if (auto store = dyn_cast<memref::StoreOp>(sop)) {
                                get_bank_expr(store.getIndices()[rank], rewriter);
                            }
                        }
                    }
                }
            }
            unsigned old_args = op.getNumArguments();
            SmallVector<unsigned> erase_arg;
            for (unsigned arg_num = 0; arg_num < old_args; ++arg_num) {
                auto arg = op.getArgument(arg_num);
                if (auto memref = dyn_cast<MemRefType>(arg.getType())) {
                    SmallVector<bool> partition;
                    for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
                        bool flag = true;
                        for (auto &use : arg.getUses()) {
                            auto sop = use.getOwner();
                            if (auto load = dyn_cast<memref::LoadOp>(sop)) {
                                if (get_bank(load.getIndices()[rank], rewriter, factor) == -1) {
                                    flag = false;
                                    break;
                                }
                            } else if (auto store = dyn_cast<memref::StoreOp>(sop)) {
                                if (get_bank(store.getIndices()[rank], rewriter, factor) == -1) {
                                    flag = false;
                                    break;
                                }
                            } else {
                                sop->dump();
                                assert(false && "Unknown memory operation");
                            }
                        }
                        partition.push_back(flag);
                    }
                    bool flag = false;
                    for (auto p : partition) {
                        if (p) {
                            flag = true;
                            break;
                        }
                    }
                    if (flag) {
                        erase_arg.push_back(arg_num);
                        SmallVector<Value> new_array;
                        SmallVector<int64_t> new_shape;
                        unsigned size = 1;
                        for (auto pair : llvm::zip(partition, memref.getShape())) {
                            if (std::get<0>(pair)) {
                                new_shape.push_back(std::get<1>(pair)/factor);
                                size *= factor;
                            } else {
                                new_shape.push_back(std::get<1>(pair));
                            }
                        }
                        auto new_memref = MemRefType::get(new_shape, memref.getElementType());
                        for (unsigned idx = 0; idx < size; ++idx) {
                            op.insertArgument(op.getNumArguments(), new_memref, {}, op.getLoc());
                            new_array.push_back(op.getArgument(op.getNumArguments()-1));
                        }
                        struct PARTITION {Operation *op; unsigned bank;};
                        SmallVector<PARTITION> new_part;
                        for (auto &use : arg.getUses()) {
                            auto sop = use.getOwner();
                            if (auto load = dyn_cast<memref::LoadOp>(sop)) {
                                unsigned bank = 0;
                                for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
                                    if (partition[rank]) {
                                        bank = bank * factor + get_bank(load.getIndices()[rank], rewriter, factor);
                                        rewriter.setInsertionPoint(load);
                                        auto new_address = get_new_address(load.getIndices()[rank], rewriter, factor);
                                        load.getIndicesMutable()[rank] = new_address.getResult();
                                    }
                                }
                                new_part.push_back(PARTITION {load, bank});
                            } else if (auto store = dyn_cast<memref::StoreOp>(sop)) {
                                unsigned bank = 0;
                                for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
                                    if (partition[rank]) {
                                        bank = bank * factor + get_bank(store.getIndices()[rank], rewriter, factor);
                                        rewriter.setInsertionPoint(store);
                                        auto new_address = get_new_address(store.getIndices()[rank], rewriter, factor);
                                        store.getIndicesMutable()[rank] = new_address.getResult();
                                    }
                                }
                                new_part.push_back(PARTITION {store, bank});
                            }
                        }
                        for (auto part : new_part) {
                            part.op->setOperand(0, new_array[part.bank]);
                        }
                    }
                }
            }
            unsigned erased = 0;
            for (auto erase : erase_arg) {
                op.eraseArgument(erase - erased);
                erased += 1;
            }
            return success();
        }

        int factor;
    };

    struct ArrayPartitionPass : ArrayPartitionBase<ArrayPartitionPass> {
        void runOnOperation() override {
            auto funcOp = getOperation();
            RewritePatternSet Patterns(&getContext());
            Patterns.add<DesignOpPattern>(funcOp.getContext(), factor);
            if (failed(applyOpPatternsAndFold(funcOp, std::move(Patterns))))
                signalPassFailure();
        }
    };

} // namespace

namespace mlir {

    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createArrayPartitionPass() {
        return std::make_unique<ArrayPartitionPass>();
    }

} // namespace mlir
