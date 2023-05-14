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

#include "TOR/TORDialect.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#include <set>
#include <iostream>

#define DEBUG_TYPE "convert-input"

namespace {
    using namespace mlir;
    using namespace mlir::arith;
    using namespace mlir::func;

    template<typename SourceOp, typename TargetOp>
    void myReplaceOp(SourceOp op, TargetOp newOp, ConversionPatternRewriter &rewriter) {
        op.getResult().replaceAllUsesWith(newOp.getResult());
        rewriter.eraseOp(op);
    }

    struct FuncOpPattern : public OpConversionPattern<FuncOp> {
        FuncOpPattern(MLIRContext *ctx, std::string top_function, std::string resource, double clock)
                : OpConversionPattern<FuncOp>(ctx), top_function(top_function), resource(resource), clock(clock)  {}

        LogicalResult
        matchAndRewrite(FuncOp op, FuncOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            rewriter.setInsertionPoint(op);
            StringRef function_name = op.getName() == top_function ? "main" : op.getName();
            auto funcOp = rewriter.create<tor::FuncOp>(op.getLoc(), function_name,
                    op.getFunctionType());
            if (function_name == "main") {
                funcOp->setAttr("clock", FloatAttr::get(FloatType::getF32(getContext()), clock));
                funcOp->setAttr("resource", StringAttr::get(getContext(), resource));
            }
            rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());
            // rewriter.setInsertionPoint(&(funcOp.getBody().front()));
            rewriter.setInsertionPointToStart(op->getBlock());
            SmallVector<Type, 4> argTypes;
            unsigned idx = 0;
            while (idx < funcOp.getNumArguments()) {
                auto arg = funcOp.getArgument(idx);
                if (auto memref = dyn_cast<MemRefType>(arg.getType())) {
                    bool R = false;
                    bool W = false;
                    std::string RW;
                    for (auto &use : arg.getUses()) {
                        if (isa<memref::LoadOp>(use.getOwner())) {
                            R = true;
                        } else if (isa<memref::StoreOp>(use.getOwner())) {
                            W = true;
                        } else {
                            assert(false);
                        }
                    }
                    if (R) RW += "r";
                    if (W) RW += "w";
                    auto shape = memref.getShape();
                    int64_t size = 1;
                    SmallVector<int64_t> one;
                    for (auto s : shape) {
                        size *= s;
                    }
                    one.push_back(size);
                    for (auto &use : arg.getUses()) {
                        if (auto load = dyn_cast<memref::LoadOp>(use.getOwner())) {
                            auto lastInsertion = rewriter.saveInsertionPoint();
                            rewriter.setInsertionPoint(load);
                            auto last_idx = load.getIndices()[0];
                            for (unsigned i = 1; i < shape.size(); ++i) {
                                int length = shape[i];
                                if (!(length & (length-1))) {
                                    int width = log2(length);
                                    auto size = rewriter.create<ConstantIndexOp>(load.getLoc(), width);
                                    auto mul = rewriter.create<ShLIOp>(load.getLoc(), last_idx, size.getResult());
                                    auto add = rewriter.create<OrIOp>(load.getLoc(), load.getIndices()[i], mul.getResult());
                                    last_idx = add.getResult();
                                } else {
                                    auto size = rewriter.create<ConstantIndexOp>(load.getLoc(), length);
                                    auto mul = rewriter.create<MulIOp>(load.getLoc(), last_idx, size.getResult());
                                    auto add = rewriter.create<AddIOp>(load.getLoc(), load.getIndices()[i], mul.getResult());
                                    last_idx = add.getResult();
                                }
                            }
                            load->setOperand(1, last_idx);
                            for (unsigned i = 1; i < shape.size(); ++i) {
                                load->eraseOperand(i+1);
                            }
                            rewriter.restoreInsertionPoint(lastInsertion);
                        } else if (auto store = dyn_cast<memref::StoreOp>(use.getOwner())) {
                            auto lastInsertion = rewriter.saveInsertionPoint();
                            rewriter.setInsertionPoint(store);
                            auto last_idx = store.getIndices()[0];
                            for (unsigned i = 1; i < shape.size(); ++i) {
                                int length = shape[i];
                                if (!(length & (length-1))) {
                                    int width = log2(length);
                                    auto size = rewriter.create<ConstantIndexOp>(store.getLoc(), width);
                                    auto mul = rewriter.create<ShLIOp>(store.getLoc(), last_idx, size.getResult());
                                    auto add = rewriter.create<OrIOp>(store.getLoc(), store.getIndices()[i], mul.getResult());
                                    last_idx = add.getResult();
                                } else {
                                    auto size = rewriter.create<ConstantIndexOp>(store.getLoc(), length);
                                    auto mul = rewriter.create<MulIOp>(store.getLoc(), last_idx, size.getResult());
                                    auto add = rewriter.create<AddIOp>(store.getLoc(), store.getIndices()[i], mul.getResult());
                                    last_idx = add.getResult();
                                }
                            }
                            store->setOperand(2, last_idx);
                            for (unsigned i = 1; i < shape.size(); ++i) {
                                store->eraseOperand(i+2);
                            }
                            rewriter.restoreInsertionPoint(lastInsertion);
                        }
                    }
                    auto newType = tor::MemRefType::get(one, memref.getElementType(), {}, StringAttr::get(getContext(), RW));
                    auto alloc = rewriter.create<tor::AllocOp>(op.getLoc(), newType);
                    arg.replaceAllUsesWith(alloc.getResult());
                    funcOp.eraseArgument(idx);
                } else {
                    argTypes.push_back(arg.getType());
                    idx += 1;
                }
            }
            rewriter.eraseOp(op);
            return success();
        }

        std::string top_function, resource;
        double clock;
    };

    struct ReturnOpConversion : public OpConversionPattern<ReturnOp> {
        using OpConversionPattern<ReturnOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(ReturnOp op, ReturnOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            rewriter.setInsertionPoint(op);
            rewriter.create<tor::ReturnOp>(op.getLoc(), adaptor.getOperands());
            rewriter.eraseOp(op);

            return success();
        }
    };

    struct LoadOpConversion : public OpConversionPattern<memref::LoadOp> {
        using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(memref::LoadOp op, memref::LoadOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            rewriter.setInsertionPoint(op);
            myReplaceOp(op, rewriter.create<tor::LoadOp>(op.getLoc(), op.getResult().getType(), adaptor.getMemref(), 0, 0, adaptor.getIndices()),
                    rewriter);

            return success();
        }
    };

    struct StoreOpConversion : public OpConversionPattern<memref::StoreOp> {
        using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(memref::StoreOp op, memref::StoreOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            rewriter.setInsertionPoint(op);
            rewriter.create<tor::StoreOp>(op.getLoc(), adaptor.getValue(), adaptor.getMemref(), 0, 0, adaptor.getIndices());
            rewriter.eraseOp(op);

            return success();
        }
    };

    struct DesignOpPattern : OpRewritePattern<mlir::ModuleOp> {
        DesignOpPattern(MLIRContext *ctx, std::string top_function)
                : OpRewritePattern<mlir::ModuleOp>(ctx), top_function(top_function) {}

        LogicalResult matchAndRewrite(mlir::ModuleOp op,
                                      PatternRewriter &rewriter) const override {
            if (isa<tor::DesignOp>(op.getBody()->front()))
                return failure();

            Region tmp_region;
            rewriter.inlineRegionBefore(op.getRegion(), tmp_region, tmp_region.begin());
            op.getRegion().push_back(new Block);
            rewriter.setInsertionPointToStart(op.getBody());
            auto designOp = rewriter.create<tor::DesignOp>(op.getLoc(), top_function);
            rewriter.inlineRegionBefore(tmp_region, designOp.getBody(), designOp.getBody().begin());
            return success();
        }

        std::string top_function;
    };

    struct ConvertInputPass : ConvertInputBase<ConvertInputPass> {
        void runOnOperation() override {
            auto moduleOp = getOperation();
            moduleOp->setAttrs(DictionaryAttr::getWithSorted(&getContext(), {}));

            {
                    RewritePatternSet Patterns(&getContext());
                    Patterns.add<DesignOpPattern>(moduleOp.getContext(), top_function);
                    if (failed(applyOpPatternsAndFold(moduleOp, std::move(Patterns))))
                        signalPassFailure();
            }

            {
                ConversionTarget target(getContext());
                RewritePatternSet patterns(&getContext());
                target.addLegalDialect<tor::TORDialect>();
                target.addLegalDialect<scf::SCFDialect>();
                target.addLegalDialect<ArithDialect>();
                target.addIllegalOp<FuncOp>();
                target.addIllegalOp<ReturnOp>();
                target.addIllegalOp<memref::LoadOp>();
                target.addIllegalOp<memref::StoreOp>();
                patterns.add<FuncOpPattern>(&getContext(), top_function, resource, clock);
                patterns.add<ReturnOpConversion, LoadOpConversion, StoreOpConversion>(&getContext());
                if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
                        signalPassFailure();
            }
        }
    };

} // namespace

namespace mlir {

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertInputPass() {
        return std::make_unique<ConvertInputPass>();
    }

} // namespace mlir
