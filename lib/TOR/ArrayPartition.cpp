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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
            auto new_map = AffineMap::get(arg_num.size(), 0, get_bank_expr(apply.getMapOperands()[0], rewriter));
            // apply.dump();
            // map.dump();
            // new_map.dump();
            // map.getResult(0).compose(new_map).dump();
            return map.getResult(0).compose(new_map);
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

    int get_bank(Value idx, PatternRewriter &rewriter, int factor, bool cyclic) {
        auto expr = get_bank_expr(idx, rewriter);
        if (!expr) {
            return -1;
        }
        if (cyclic) {
            expr = expr % factor;
        } else {
            expr = expr.floorDiv(factor);
        }
        auto map = AffineMap::get(arg_num.size(), 0, expr);
        if (map.isConstant()) {
            return map.getConstantResults()[0];
        }
        return -1;
    }

    Operation *get_new_address(Operation *op, Value idx, PatternRewriter &rewriter, int factor, bool cyclic) {
        arg_num.clear();
        auto expr = cyclic ? getAffineDimExpr(0, rewriter.getContext()).floorDiv(factor) :
            getAffineDimExpr(0, rewriter.getContext()) % factor;
        auto new_map = AffineMap::get(arg_num.size(), 0, get_bank_expr(idx, rewriter));
        auto map = AffineMap::get(arg_num.size(), 0, expr.compose(new_map));
        if (arg_num.empty()) {
            assert(map.isConstant());
            rewriter.setInsertionPoint(op);
            return rewriter.create<arith::ConstantIndexOp>(op->getLoc(), map.getConstantResults()[0]);
        }
        SmallVector<Value> apply;
        for (unsigned i = 0; i < arg_num.size(); ++i) {
            apply.push_back(Value());
        }
        for (auto pair : arg_num) {
            apply[pair.second] = Value(pair.first);
        }
        rewriter.setInsertionPoint(op);
        return rewriter.create<AffineApplyOp>(op->getLoc(), map, apply);
    }

    struct DesignOpPattern : OpRewritePattern<FuncOp> {
        DesignOpPattern(MLIRContext *ctx, int *factor, bool *cyclic, int arg_num)
                : OpRewritePattern<FuncOp>(ctx), factor(factor), cyclic(cyclic), arg_num(arg_num) {}

        LogicalResult matchAndRewrite(FuncOp op,
                                      PatternRewriter &rewriter) const override {
            if (op->hasAttr("array-partition"))
                return failure();
            op->setAttr("array-partition", IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
            
            auto arg = op.getArgument(arg_num);
            auto memref = cast<MemRefType>(arg.getType());
            // for (int rank = 0; rank < memref.getRank(); ++rank) {
            //     if (rank == dimension) {
            //         for (auto &use : arg.getUses()) {
            //             auto sop = use.getOwner();
            //             sop->dump();
            //             if (auto load = dyn_cast<AffineLoadOp>(sop)) {
            //                 get_bank_expr(load.getIndices()[rank], rewriter);
            //             } else if (auto store = dyn_cast<AffineStoreOp>(sop)) {
            //                 get_bank_expr(store.getIndices()[rank], rewriter);
            //             }
            //         }
            //     }
            // }
            SmallVector<bool> partition;
            for (int rank = 0; rank < memref.getRank(); ++rank) {
                bool flag = true;
                if (factor[rank] > 1) {
                    for (auto &use : arg.getUses()) {
                        auto sop = use.getOwner();
                        if (auto load = dyn_cast<AffineLoadOp>(sop)) {
                            unsigned bank_factor = cyclic[rank] ? factor[rank] : memref.getShape()[rank] / factor[rank];
                            if (get_bank(load.getIndices()[rank], rewriter, bank_factor, cyclic[rank]) == -1) {
                                flag = false;
                                break;
                            }
                        } else if (auto store = dyn_cast<AffineStoreOp>(sop)) {
                            unsigned bank_factor = cyclic[rank] ? factor[rank] : memref.getShape()[rank] / factor[rank];
                            if (get_bank(store.getIndices()[rank], rewriter, bank_factor, cyclic[rank]) == -1) {
                                flag = false;
                                break;
                            }
                        } else {
                            sop->dump();
                            assert(false && "Unknown memory operation");
                        }
                    }                    
                } else {
                    flag = false;
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
                arg.dump();
                for (auto p : partition) {
                    std::cerr<<p<<" ";
                }
                std::cerr<<std::endl;
                SmallVector<Value> new_array;
                SmallVector<int64_t> new_shape;
                unsigned size = 1;
                for (int rank = 0; rank < memref.getRank(); ++rank) {
                    if (partition[rank]) {
                        new_shape.push_back(memref.getShape()[rank]/factor[rank]);
                        size *= factor[rank];
                    } else {
                        new_shape.push_back(memref.getShape()[rank]);
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
                    if (auto load = dyn_cast<AffineLoadOp>(sop)) {
                        SmallVector<Value> idx = load.getIndices();
                        unsigned bank = 0;
                        for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
                            if (partition[rank]) {
                                unsigned bank_factor = cyclic[rank] ? factor[rank] : memref.getShape()[rank] / factor[rank];
                                bank = bank * factor[rank] + get_bank(idx[rank], rewriter, bank_factor, cyclic[rank]);
                                auto new_address = get_new_address(load, idx[rank], rewriter, bank_factor, cyclic[rank]);
                                idx[rank] = new_address->getResult(0);
                            }
                        }
                        for (unsigned i = 0; i < idx.size(); ++i) {
                            load->setOperand(i+1, idx[i]);
                        }
                        new_part.push_back(PARTITION {load, bank});
                    } else if (auto store = dyn_cast<AffineStoreOp>(sop)) {
                        SmallVector<Value> idx = store.getIndices();
                        unsigned bank = 0;
                        for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
                            if (partition[rank]) {
                                unsigned bank_factor = cyclic[rank] ? factor[rank] : memref.getShape()[rank] / factor[rank];
                                bank = bank * factor[rank] + get_bank(idx[rank], rewriter, bank_factor, cyclic[rank]);
                                auto new_address = get_new_address(store, idx[rank], rewriter, bank_factor, cyclic[rank]);
                                idx[rank] = new_address->getResult(0);
                            }
                        }
                        for (unsigned i = 0; i < idx.size(); ++i) {
                            store->setOperand(i+2, idx[i]);
                        }
                        new_part.push_back(PARTITION {store, bank});
                    }
                }
                for (auto part : new_part) {
                    part.op->setOperand(isa<AffineStoreOp>(part.op), new_array[part.bank]);
                }
                op.eraseArgument(arg_num);
            }
            return success();
        }

        int *factor;
        bool *cyclic;
        int arg_num;
    };

    struct ArrayPartitionPass : ArrayPartitionBase<ArrayPartitionPass> {
        void runOnOperation() override {
            auto funcOp = getOperation();
            RewritePatternSet Patterns(&getContext());
            int factor_vec[10];
            for (unsigned i=0; i<factor.size(); ++i) {
                factor_vec[i] = factor[i];
            }
            bool cyclic_vec[10];
            for (unsigned i=0; i<cyclic.size(); ++i) {
                cyclic_vec[i] = cyclic[i];
            }
            Patterns.add<DesignOpPattern>(funcOp.getContext(), factor_vec, cyclic_vec, arg_num);
            if (failed(applyOpPatternsAndFold(funcOp, std::move(Patterns))))
                signalPassFailure();
            funcOp->removeAttr("array-partition");
        }
    };

} // namespace

namespace mlir {

    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createArrayPartitionPass() {
        return std::make_unique<ArrayPartitionPass>();
    }

} // namespace mlir
