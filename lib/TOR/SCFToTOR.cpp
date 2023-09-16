#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

#define DEBUG_TYPE "scf-to-tor"

namespace {
    using namespace mlir;

    struct ConstIndexConversion : public OpConversionPattern<ConstantOp> {
        using OpConversionPattern<ConstantOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(ConstantOp op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {

            auto value = op.getValue();
            rewriter.setInsertionPoint(op);
            if (op.getResult().getType().isa<IndexType>()) {
                auto newOp = rewriter.create<ConstantIntOp>(
                        op->getLoc(), value.cast<IntegerAttr>().getInt(), 32);
                newOp->setAttr("dump", op->getAttr("dump"));
                rewriter.replaceOp(op, newOp.getResult());
                return success();
            }

            return failure();
        }
    };

    struct YieldOpConversion : public OpConversionPattern<scf::YieldOp> {
        using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(scf::YieldOp op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
            for (auto opr : operands)
                if (opr.getType().isa<IndexType>())
                    return failure();

            rewriter.setInsertionPoint(op);
            rewriter.create<tor::YieldOp>(op.getLoc(), operands);
            rewriter.eraseOp(op);

            return success();
        }
    };

    struct IfOpConversion : public OpConversionPattern<scf::IfOp> {
        using OpConversionPattern<scf::IfOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(scf::IfOp op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
            for (auto opr : operands)
                if (opr.getType().isa<IndexType>())
                    return failure();

            rewriter.setInsertionPoint(op);

            SmallVector<Type, 4> resultTypes(op.getResultTypes());

            for (auto &type : resultTypes)
                if (type.isa<IndexType>())
                    type = IntegerType::get(getContext(), 32);

            auto newOp = rewriter.create<tor::IfOp>(op.getLoc(), resultTypes,
                                                    operands[0], 0, 0);
            newOp->setAttr("dump", op->getAttr("dump"));

            rewriter.createBlock(&newOp.thenRegion());
            rewriter.inlineRegionBefore(op.thenRegion(), &newOp.thenRegion().back());
            rewriter.eraseBlock(&newOp.thenRegion().back());

            if (!op.elseRegion().empty()) {
                rewriter.createBlock(&newOp.elseRegion());
                rewriter.inlineRegionBefore(op.elseRegion(), &newOp.elseRegion().back());
                rewriter.eraseBlock(&newOp.elseRegion().back());
            }

            rewriter.replaceOp(op, newOp.getResults());

            return success();
        }
    };

    struct CondOpConversion : public OpConversionPattern<scf::ConditionOp> {
        using OpConversionPattern<scf::ConditionOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(scf::ConditionOp op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
            for (auto opr : operands)
                if (opr.getType().isa<IndexType>())
                    return failure();

            rewriter.create<tor::ConditionOp>(op.getLoc(), operands[0],
                                              operands.drop_front(1));

            rewriter.eraseOp(op);

            return success();
        }
    };

    struct WhileOpConversion : public OpConversionPattern<scf::WhileOp> {
        using OpConversionPattern<scf::WhileOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(scf::WhileOp op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
            for (auto opr : operands)
                if (opr.getType().isa<IndexType>())
                    return failure();

            SmallVector<Type, 4> resultTypes(op.getResultTypes());
            for (auto &type : resultTypes)
                if (type.isa<IndexType>())
                    type = IntegerType::get(getContext(), 32);

            rewriter.setInsertionPoint(op);
            auto newOp = rewriter.create<tor::WhileOp>(op.getLoc(), resultTypes,
                                                       operands, 0, 0);

            newOp->setAttrs(op->getAttrDictionary());
            newOp->setAttr("starttime",
                           mlir::IntegerAttr::get(
                                   mlir::IntegerType::get(getContext(), 32,
                                                          mlir::IntegerType::Signless),
                                   0));
            newOp->setAttr("endtime",
                           mlir::IntegerAttr::get(
                                   mlir::IntegerType::get(getContext(), 32,
                                                          mlir::IntegerType::Signless),
                                   0));

            rewriter.inlineRegionBefore(op.before(), newOp.before(),
                                        newOp.before().begin());
            rewriter.inlineRegionBefore(op.after(), newOp.after(),
                                        newOp.after().begin());
            rewriter.replaceOp(op, newOp.getResults());

            return success();
        }
    };

    struct ForOpConversion : public OpConversionPattern<scf::ForOp> {
        using OpConversionPattern<scf::ForOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(scf::ForOp op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
            for (auto opr : operands)
                if (opr.getType().isa<IndexType>())
                    return failure();

            rewriter.setInsertionPoint(op);
            auto newOp = rewriter.create<tor::ForOp>(
                    op.getLoc(), operands[0], operands[1], operands[2],
                    mlir::IntegerAttr::get(
                            mlir::IntegerType::get(getContext(), 32,
                                                   mlir::IntegerType::Signless),
                            0),
                    mlir::IntegerAttr::get(
                            mlir::IntegerType::get(getContext(), 32,
                                                   mlir::IntegerType::Signless),
                            0),
                    ValueRange(operands.drop_front(3)));

            newOp->setAttrs(op->getAttrDictionary());
            newOp->setAttr("starttime",
                           mlir::IntegerAttr::get(
                                   mlir::IntegerType::get(getContext(), 32,
                                                          mlir::IntegerType::Signless),
                                   0));
            newOp->setAttr("endtime",
                           mlir::IntegerAttr::get(
                                   mlir::IntegerType::get(getContext(), 32,
                                                          mlir::IntegerType::Signless),
                                   0));

            rewriter.inlineRegionBefore(op.region(), newOp.region(),
                                        newOp.region().begin());

            for (auto pair : llvm::zip(newOp.getBody()->getArguments(),
                                       newOp.getBody()->getArgumentTypes()))
                if (std::get<1>(pair).isa<IndexType>())
                    std::get<0>(pair).setType(IntegerType::get(getContext(), 32));

            rewriter.replaceOp(op, newOp.getResults());

            return success();
        }
    };

    struct CallOpConversion : public OpConversionPattern<mlir::CallOp> {
        using OpConversionPattern<CallOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(mlir::CallOp op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
            for (auto opr : operands)
                if (opr.getType().isa<IndexType>())
                    return failure();

            rewriter.setInsertionPoint(op);
            auto newOp = rewriter.create<tor::CallOp>(op.getLoc(), op.getResultTypes(),
                                                      op.getCallee(), 0, 0, operands);
            newOp->setAttr("dump", op->getAttr("dump"));
            rewriter.replaceOp(op, newOp.getResults());
            return success();
        }
    };

    template<typename SourceOp, typename TargetOp>
    struct BinIOpConversion : public OpConversionPattern<SourceOp> {
        using OpConversionPattern<SourceOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
            assert(operands.size() == 2 && "addi has two operand");

            for (auto opr : operands)
                if (opr.getType().isa<IndexType>())
                    return failure();

            rewriter.setInsertionPoint(op);

            TargetOp newOp;

            if (op.getResult().getType().template isa<IndexType>())
                newOp = rewriter.create<TargetOp>(op.getLoc(), operands[0], operands[1]);
            else
                newOp = rewriter.create<TargetOp>(op.getLoc(), op.getResult().getType(),
                                                  operands[0], operands[1], 0, 0);
            newOp->setAttr("dump", op->getAttr("dump"));

            rewriter.replaceOp(op, newOp.getResult());

            return success();
        }
    };

    template<typename SourceOp, typename TargetOp>
    struct BinFOpConversion : public OpConversionPattern<SourceOp> {
        using OpConversionPattern<SourceOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
            rewriter.setInsertionPoint(op);

            TargetOp newOp = rewriter.create<TargetOp>(
                    op.getLoc(), op.getResult().getType(), operands[0], operands[1], 0, 0);
            newOp->setAttr("dump", op->getAttr("dump"));

            rewriter.replaceOp(op, newOp.getResult());

            return success();
        }
    };

    using MulIOpConversion = BinIOpConversion<MulIOp, tor::MulIOp>;
    using AddIOpConversion = BinIOpConversion<AddIOp, tor::AddIOp>;
    using SubIOpConversion = BinIOpConversion<SubIOp, tor::SubIOp>;
    using MulFOpConversion = BinFOpConversion<MulFOp, tor::MulFOp>;
    using AddFOpConversion = BinFOpConversion<AddFOp, tor::AddFOp>;
    using SubFOpConversion = BinFOpConversion<SubFOp, tor::SubFOp>;
    using DivFOpConversion = BinFOpConversion<DivFOp, tor::DivFOp>;

    struct CmpIOpConversion : public OpConversionPattern<CmpIOp> {
        using OpConversionPattern<CmpIOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(CmpIOp op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
            assert(operands.size() == 2 && "addi has two operand");

            for (auto opr : operands)
                if (opr.getType().isa<IndexType>())
                    return failure();

            rewriter.setInsertionPoint(op);
            auto predicate = static_cast<mlir::tor::CmpIPredicate>(op.predicate());
            auto newOp = rewriter.create<tor::CmpIOp>(
                    op.getLoc(), op.getResult().getType(), predicate, operands[0],
                    operands[1], 0, 0);
            newOp->setAttr("dump", op->getAttr("dump"));

            rewriter.replaceOp(op, newOp.getResult());

            return success();
        }
    };

    struct CmpFOpConversion : public OpConversionPattern<CmpFOp> {
        using OpConversionPattern<CmpFOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(CmpFOp op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
            assert(operands.size() == 2 && "cmpf has two operand");

            rewriter.setInsertionPoint(op);
            auto predicate = static_cast<mlir::tor::CmpFPredicate>(op.predicate());
            auto newOp = rewriter.create<tor::CmpFOp>(
                    op.getLoc(), op.getResult().getType(), predicate, operands[0],
                    operands[1], 0, 0);
            newOp->setAttr("dump", op->getAttr("dump"));

            rewriter.replaceOp(op, newOp.getResult());

            return success();
        }
    };

    struct CastOpErasure : public OpConversionPattern<IndexCastOp> {
        using OpConversionPattern<IndexCastOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(IndexCastOp op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
            if (operands[0].getType().isa<IndexType>())
                return failure();

            rewriter.replaceOp(op, operands);
            return success();
        }
    };

    struct ShiftLeftConversionPattern : public OpConversionPattern<ShiftLeftOp> {
        using OpConversionPattern<ShiftLeftOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(ShiftLeftOp op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
            for (auto opr : operands)
                if (opr.getType().isa<IndexType>())
                    return failure();
            if (!op.getResult().getType().isa<IndexType>())
                return failure();

            auto newOp =
                    rewriter.create<ShiftLeftOp>(op.getLoc(), operands[0], operands[1]);
            newOp->setAttr("dump", op->getAttr("dump"));

            rewriter.replaceOp(op, newOp.getResult());
            llvm::outs() << "Yeh!\n";
            return success();
        }
    };

    struct FuncArgCovnersion : public OpConversionPattern<tor::FuncOp> {
        using OpConversionPattern<tor::FuncOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(tor::FuncOp op, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
            SmallVector<Type, 4> newInputTypes;
            for (auto type : op.getType().getInputs())
                if (type.isa<IndexType>())
                    newInputTypes.push_back(IntegerType::get(getContext(), 32));
                else
                    newInputTypes.push_back(type);

            rewriter.updateRootInPlace(op, [&] {
                for (auto arg : op.getArguments())
                    if (arg.getType().isa<IndexType>())
                        arg.setType(IntegerType::get(getContext(), 32));
                op.setType(FunctionType::get(getContext(), newInputTypes,
                                             op.getType().getResults()));
            });

            return success();
        }
    };

    void IterativeConstantFolding(mlir::tor::FuncOp funcOp,
                                  PatternRewriter &rewriter) {
        std::set<mlir::Operation *> WorkingList;

        //  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
        funcOp.walk([&](mlir::Operation *op) -> mlir::WalkResult {
            WorkingList.insert(op);
            return mlir::WalkResult::advance();
        });

        while (!WorkingList.empty()) {
            auto op = *WorkingList.begin();
            WorkingList.erase(op);
            mlir::SmallVector<mlir::Value> results;
            // Special Case because of chisel module
            if (mlir::succeeded(rewriter.tryFold(op, results))) {
                rewriter.replaceOp(op, results);
                for (auto value : results)
                    for (auto succop : value.getUsers())
                        WorkingList.insert(succop);
            } else if (auto sitofOp = llvm::dyn_cast<mlir::SIToFPOp>(op)) {
                mlir::APInt val;
                if (mlir::matchPattern(sitofOp.getOperand(), mlir::m_ConstantInt(&val))) {
                    mlir::Operation *op;
                    if (sitofOp.getResult().getType().isF32())
                        op = rewriter.create<mlir::ConstantFloatOp>(
                                sitofOp.getLoc(), mlir::APFloat((float) val.roundToDouble()),
                                sitofOp.getResult().getType().cast<mlir::FloatType>());
                    else
                        op = rewriter.create<mlir::ConstantFloatOp>(
                                sitofOp.getLoc(), mlir::APFloat(val.roundToDouble()),
                                sitofOp.getResult().getType().cast<mlir::FloatType>());

                    auto constOp = llvm::dyn_cast<mlir::ConstantOp>(op);
                    rewriter.replaceOp(sitofOp, constOp.getResult());
                    for (auto succop : constOp.getResult().getUsers())
                        WorkingList.insert(succop);
                }
            }
        }
    }

    struct ConstantFoldingPattern : OpRewritePattern<tor::FuncOp> {
        using OpRewritePattern<tor::FuncOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(tor::FuncOp op,
                                      PatternRewriter &rewriter) const override {
            if (op->getAttr("constant-folded"))
                return failure();
            IterativeConstantFolding(op, rewriter);
            op->setAttr("constant-folded",
                        IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
            return success();
        }
    };

    struct MoveConstantUp : OpRewritePattern<ConstantOp> {
        using OpRewritePattern<ConstantOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ConstantOp op, PatternRewriter &rewriter) const override {
            if (llvm::isa<tor::DesignOp>(op->getParentOp()))
                return failure();

            auto topParent = op->getParentOfType<tor::DesignOp>();

            assert(topParent);

            rewriter.setInsertionPoint(topParent.getBody(),
                                       topParent.getBody()->begin());

            auto newOp = rewriter.clone(*op.getOperation());

            rewriter.replaceOp(op, newOp->getResults());

            return success();
        }
    };

    struct SCFToTORPass : SCFToTORBase<SCFToTORPass> {
        void runOnOperation() override {
            auto designOp = getOperation();

            {
                designOp.walk([&](tor::FuncOp op) {
                    RewritePatternSet rPatterns(&getContext());
                    rPatterns.insert<ConstantFoldingPattern>(&getContext());
                    if (failed(applyOpPatternsAndFold(op, std::move(rPatterns))))
                        WalkResult::interrupt();
                    WalkResult::advance();
                });
            }

            {
                ConversionTarget target(getContext());
                RewritePatternSet patterns(&getContext());

                target.addLegalDialect<tor::TORDialect>();
                target.addDynamicallyLegalOp<ShiftLeftOp>([](ShiftLeftOp op) {
                    llvm::outs() << "GOOD\n";
                    if (op.getResult().getType().isa<IndexType>())
                        return false;
                    return true;
                });
                target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp op) {
                    if (op.getResult().getType().isa<IndexType>())
                        return false;
                    if (!llvm::isa<tor::DesignOp>(op->getParentOp()))
                        return false;
                    return true;
                });
                target.addDynamicallyLegalOp<tor::FuncOp>([](tor::FuncOp op) {
                    for (auto type : op.getArgumentTypes())
                        if (type.isa<IndexType>())
                            return false;
                    return true;
                });

                patterns.add<AddIOpConversion, ConstIndexConversion, MulIOpConversion,
                        SubIOpConversion, CmpIOpConversion, MulFOpConversion,
                        AddFOpConversion, SubFOpConversion, DivFOpConversion,
                        YieldOpConversion, CondOpConversion, WhileOpConversion,
                        ForOpConversion, IfOpConversion, FuncArgCovnersion,
                        CastOpErasure, CmpFOpConversion, ShiftLeftConversionPattern,
                        MoveConstantUp>(&getContext());

                if (failed(applyPartialConversion(designOp, target, std::move(patterns))))
                    signalPassFailure();
            }

        }
    };

} // namespace

namespace mlir {

    std::unique_ptr<OperationPass<tor::DesignOp>> createSCFToTORPass() {
        return std::make_unique<SCFToTORPass>();
    }

} // namespace mlir
