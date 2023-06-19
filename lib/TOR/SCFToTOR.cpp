#include "TOR/TOR.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"

#include "TOR/TORDialect.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"
#include "llvm/ADT/STLExtras.h"

#include <set>
#include <iostream>
#include <type_traits>

#define DEBUG_TYPE "scf-to-tor"

namespace {
    using namespace mlir;
    using namespace mlir::arith;

    std::string get_opertion_attr() {
        static int attr_num = 0;
        return "control_" + std::to_string(attr_num++);
    }

    std::string get_tmp_attr() {
        static int attr_num = 0;
        return "unknown_" + std::to_string(attr_num++);
    }

    template<typename SourceOp, typename TargetOp>
    void myReplaceOp(SourceOp op, TargetOp newOp, ConversionPatternRewriter &rewriter) {
        // for (auto [newValue, oldValue] : llvm::zip(op->getResults(), newOp->getResults())) {
        //     newValue.replaceAllUsesWith(oldValue);
        // }
        rewriter.replaceOp(op, newOp->getResults());
        // op.getResult().replaceAllUsesWith(newOp.getResult());
        // rewriter.eraseOp(op);
    }

    template<typename SourceOp>
    struct IndexTypeConversionPattern: public OpConversionPattern<SourceOp> {
        using OpConversionPattern<SourceOp>::OpConversionPattern;
        SmallVector<Value> prepareOperands(SourceOp op, typename SourceOp::Adaptor adaptor, ConversionPatternRewriter & rewriter) const {
            auto operands = adaptor.getOperands();
            SmallVector<Type> newOperandTypes;
            auto converter = this->getTypeConverter();
            (void) converter->convertTypes(op->getOperandTypes(), newOperandTypes);
            return llvm::to_vector(llvm::map_range(llvm::zip(operands, newOperandTypes), [&](auto it) {
                auto [operand, tpe] = it;
                return converter->materializeSourceConversion(rewriter, op->getLoc(), tpe, ValueRange(operand));
            }));
        }
    };
    struct YieldOpConversion : public IndexTypeConversionPattern<scf::YieldOp> {
        using IndexTypeConversionPattern<scf::YieldOp>::IndexTypeConversionPattern;
        LogicalResult matchAndRewrite(scf::YieldOp op, typename scf::YieldOp::Adaptor adaptor, ConversionPatternRewriter & rewriter) const final {
            auto operands = this->prepareOperands(op, adaptor, rewriter);
            auto newOp = rewriter.replaceOpWithNewOp<tor::YieldOp>(op, operands);
            newOp->setAttr("dump", op->getAttr("dump"));
            return success();
        }
    };
    struct CondOpConversion : public IndexTypeConversionPattern<scf::ConditionOp> {
        using IndexTypeConversionPattern<scf::ConditionOp>::IndexTypeConversionPattern;
        LogicalResult matchAndRewrite(scf::ConditionOp op, typename scf::ConditionOp::Adaptor adaptor, ConversionPatternRewriter & rewriter) const final {
            auto operands = this->prepareOperands(op, adaptor, rewriter);
            auto newOp = rewriter.replaceOpWithNewOp<tor::ConditionOp>(op, operands[0], ValueRange(operands).drop_front(1));
            newOp->setAttr("dump", op->getAttr("dump"));
            return success();
        }
    };
    template<typename SourceOp, typename TargetOp>
    struct BinOpConversionPattern: public IndexTypeConversionPattern<SourceOp> {
        using IndexTypeConversionPattern<SourceOp>::IndexTypeConversionPattern;
        LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor, ConversionPatternRewriter & rewriter) const final {
            auto operands = this->prepareOperands(op, adaptor, rewriter);
            auto resType = this->getTypeConverter()->convertType(op->getResult(0).getType());
            auto newOp = rewriter.replaceOpWithNewOp<TargetOp>(op, resType, operands[0], operands[1], 0, 0);
            newOp->setAttr("dump", op->getAttr("dump"));
            return success();
        }
    };
    using MulIOpConversion  = BinOpConversionPattern<MulIOp, tor::MulIOp>;
    using AddIOpConversion  = BinOpConversionPattern<AddIOp, tor::AddIOp>;
    using SubIOpConversion  = BinOpConversionPattern<SubIOp, tor::SubIOp>;
    using MulFOpConversion  = BinOpConversionPattern<MulFOp, tor::MulFOp>;
    using AddFOpConversion  = BinOpConversionPattern<AddFOp, tor::AddFOp>;
    using SubFOpConversion  = BinOpConversionPattern<SubFOp, tor::SubFOp>;
    using DivFOpConversion  = BinOpConversionPattern<DivFOp, tor::DivFOp>;
    template<typename SourceOp>
    struct SimpleOpConversion: public IndexTypeConversionPattern<SourceOp> {
        using IndexTypeConversionPattern<SourceOp>::IndexTypeConversionPattern;
        LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor, ConversionPatternRewriter & rewriter) const final {
            if(!llvm::any_of(op.getOperandTypes(), [](auto tpe){return isa<IndexType>(tpe);}))
                return failure();
            auto operands = this->prepareOperands(op, adaptor, rewriter);
            SmallVector<Type> resultTypes;
            (void) this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes);
            rewriter.replaceOpWithNewOp<SourceOp>(op, resultTypes, operands, op->getAttrs());
            return success();
        }
    };
    using ShiftLeftConversionPattern = SimpleOpConversion<ShLIOp>;
    using OrIConversionPattern = SimpleOpConversion<OrIOp>;
    using SelectConversionPattern = SimpleOpConversion<SelectOp>;
    using LoadOpConversion = SimpleOpConversion<tor::LoadOp>;
    using StoreOpConversion = SimpleOpConversion<tor::StoreOp>;

    struct ConstIndexConversion : public OpConversionPattern<ConstantOp> {
        using OpConversionPattern<ConstantOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(ConstantOp op, ConstantOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            if (op.getResult().getType().isa<IndexType>()) {
                auto value = adaptor.getValue();
                rewriter.setInsertionPoint(op);
                auto newOp = rewriter.create<ConstantIntOp>(
                        op->getLoc(), value.cast<IntegerAttr>().getInt(), 32);
                //FIXME: ???
                if (!op->hasAttr("dump")) {
                    op->setAttr("dump", StringAttr::get(rewriter.getContext(), get_tmp_attr().c_str()));
                    // op->dump();
                }
                newOp->setAttr("dump", op->getAttr("dump"));
                // rewriter.replaceOp(op, {newOp.getResult()});
                myReplaceOp(op, newOp, rewriter);
                return success();
            }

            return failure();
        }
    };

    struct IfOpConversion : public OpConversionPattern<scf::IfOp> {
        using OpConversionPattern<scf::IfOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(scf::IfOp op, scf::IfOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            auto operands = adaptor.getOperands();
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

            rewriter.createBlock(&newOp.getThenRegion());
            rewriter.inlineRegionBefore(op.getThenRegion(), &newOp.getThenRegion().back());
            rewriter.eraseBlock(&newOp.getThenRegion().back());

            if (!op.getElseRegion().empty()) {
                rewriter.createBlock(&newOp.getElseRegion());
                rewriter.inlineRegionBefore(op.getElseRegion(), &newOp.getElseRegion().back());
                rewriter.eraseBlock(&newOp.getElseRegion().back());
            }

            rewriter.replaceOp(op, newOp.getResults());

            return success();
        }
    };

    struct WhileOpConversion : public OpConversionPattern<scf::WhileOp> {
        using OpConversionPattern<scf::WhileOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(scf::WhileOp op, scf::WhileOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            auto operands = adaptor.getOperands();
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

            rewriter.inlineRegionBefore(op.getBefore(), newOp.getBefore(),
                                        newOp.getBefore().begin());
            rewriter.inlineRegionBefore(op.getAfter(), newOp.getAfter(),
                                        newOp.getAfter().begin());
            rewriter.replaceOp(op, newOp.getResults());

            return success();
        }
    };

    struct ForOpConversion : public OpConversionPattern<scf::ForOp> {
        using OpConversionPattern<scf::ForOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(scf::ForOp op, scf::ForOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            auto operands = adaptor.getOperands();
            for (auto opr : operands)
                if (opr.getType().isa<IndexType>())
                    return failure();

            rewriter.setInsertionPoint(op);
            auto upperBound = rewriter.create<SubIOp>(op.getLoc(), operands[1], operands[2]);
            upperBound->setAttr("dump", StringAttr::get(rewriter.getContext(), get_opertion_attr().c_str()));

            auto newOp = rewriter.create<tor::ForOp>(
                    op.getLoc(), operands[0], upperBound.getResult(), operands[2],
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

            rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                        newOp.getRegion().begin());

            for (auto pair : llvm::zip(newOp.getBody()->getArguments(),
                                       newOp.getBody()->getArgumentTypes()))
                if (std::get<1>(pair).isa<IndexType>())
                    std::get<0>(pair).setType(IntegerType::get(getContext(), 32));

            rewriter.replaceOp(op, newOp.getResults());
            

            return success();
        }
    };

    struct CallOpConversion : public OpConversionPattern<func::CallOp> {
        using OpConversionPattern<func::CallOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(func::CallOp op, func::CallOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            auto operands = adaptor.getOperands();
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
    struct CmpIOpConversion : public OpConversionPattern<CmpIOp> {
        using OpConversionPattern<CmpIOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(CmpIOp op, CmpIOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            auto operands = adaptor.getOperands();
            assert(operands.size() == 2 && "addi has two operand");

            for (auto opr : operands)
                if (opr.getType().isa<IndexType>())
                    return failure();

            rewriter.setInsertionPoint(op);
            auto predicate = static_cast<mlir::tor::CmpIPredicate>(op.getPredicate());
            auto newOp = rewriter.create<tor::CmpIOp>(
                    op.getLoc(), op.getResult().getType(), predicate, operands[0],
                    operands[1], 0, 0);
            newOp->setAttr("dump", op->getAttr("dump"));

            // rewriter.replaceOp(op, newOp.getResult());
            myReplaceOp(op, newOp, rewriter);

            return success();
        }
    };

    struct CmpFOpConversion : public OpConversionPattern<CmpFOp> {
        using OpConversionPattern<CmpFOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(CmpFOp op, CmpFOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            auto operands = adaptor.getOperands();
            assert(operands.size() == 2 && "cmpf has two operand");

            rewriter.setInsertionPoint(op);
            auto predicate = static_cast<mlir::tor::CmpFPredicate>(op.getPredicate());
            auto newOp = rewriter.create<tor::CmpFOp>(
                    op.getLoc(), op.getResult().getType(), predicate, operands[0],
                    operands[1], 0, 0);
            newOp->setAttr("dump", op->getAttr("dump"));

            // rewriter.replaceOp(op, newOp.getResult());
            myReplaceOp(op, newOp, rewriter);

            return success();
        }
    };

    struct FuncOpPattern : public OpConversionPattern<tor::FuncOp> {
        using OpConversionPattern<tor::FuncOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(tor::FuncOp op, tor::FuncOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            SmallVector<Type, 4> newInputTypes;
            for (auto type : op.getFunctionType().getInputs())
                if (type.isa<IndexType>())
                    newInputTypes.push_back(IntegerType::get(getContext(), 32));
                else
                    newInputTypes.push_back(type);

            rewriter.updateRootInPlace(op, [&] {
                for (auto arg : op.getArguments())
                    if (arg.getType().isa<IndexType>())
                        arg.setType(IntegerType::get(getContext(), 32));
                op.setType(FunctionType::get(getContext(), newInputTypes,
                                             op.getFunctionType().getResults()));
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
            } else if (auto sitofOp = llvm::dyn_cast<SIToFPOp>(op)) {
                mlir::APInt val;
                if (mlir::matchPattern(sitofOp.getOperand(), mlir::m_ConstantInt(&val))) {
                    mlir::Operation *op;
                    if (sitofOp.getResult().getType().isF32())
                        op = rewriter.create<ConstantFloatOp>(
                                sitofOp.getLoc(), mlir::APFloat((float) val.roundToDouble()),
                                sitofOp.getResult().getType().cast<mlir::FloatType>());
                    else
                        op = rewriter.create<ConstantFloatOp>(
                                sitofOp.getLoc(), mlir::APFloat(val.roundToDouble()),
                                sitofOp.getResult().getType().cast<mlir::FloatType>());

                    auto constOp = llvm::dyn_cast<ConstantOp>(op);
                    rewriter.replaceOp(sitofOp, constOp.getResult());
                    for (auto succop : constOp.getResult().getUsers())
                        WorkingList.insert(succop);
                }
            }
        }
    }

    struct DesignOpPattern : OpRewritePattern<tor::FuncOp> {
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

            rewriter.setInsertionPoint(&(topParent.getBody().front()),
                                       topParent.getBody().front().begin());

            auto newOp = rewriter.clone(*op.getOperation());

            rewriter.replaceOp(op, newOp->getResults());

            return success();
        }
    };

    class IndexTypeConverter : public TypeConverter {
    public:
    IndexTypeConverter() {
        addConversion([](Type type) { return type; });
        addConversion(convertIndexType);
        auto addUnrealizedCast = [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
            auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
            return Optional<Value>(cast.getResult(0));
        };
        addSourceMaterialization(addUnrealizedCast);
        addTargetMaterialization(addUnrealizedCast);
    }
    static Optional<Type> convertIndexType(Type type) {
        if(type.isa<IndexType>()) {
            return IntegerType::get(type.getContext(), 32);
        }
        return llvm::None;
    }
    };

    struct ConvertInputPass : SCFToTORBase<ConvertInputPass> {
        void runOnOperation() override {
            auto designOp = getOperation();

            // {
            //     designOp.walk([&](tor::FuncOp op) {
            //         RewritePatternSet rPatterns(&getContext());
            //         rPatterns.insert<DesignOpPattern>(&getContext());
            //         if (failed(applyOpPatternsAndFold(op, std::move(rPatterns))))
            //             WalkResult::interrupt();
            //         WalkResult::advance();
            //     });
            // }

            {
                ConversionTarget target(getContext());
                RewritePatternSet patterns(&getContext());
                IndexTypeConverter converter;
                

                target.addLegalDialect<tor::TORDialect>();
                auto hasIndexType = [](Operation * op) {
                    if(llvm::any_of(op->getOperandTypes(), [&](auto tpe){return isa<IndexType>(tpe);})) {
                        return false;
                    }
                    if(llvm::any_of(op->getResultTypes(), [&](auto tpe){return isa<IndexType>(tpe);})) {
                        return false;
                    }
                    return true;
                };
                target.addDynamicallyLegalOp<ShLIOp>(hasIndexType);
                target.addDynamicallyLegalOp<OrIOp>(hasIndexType);
                target.addDynamicallyLegalOp<ConstantOp>(hasIndexType);
                target.addDynamicallyLegalOp<tor::LoadOp>(hasIndexType);
                target.addDynamicallyLegalOp<tor::StoreOp>(hasIndexType);
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
                        ForOpConversion, IfOpConversion, FuncOpPattern,
                        CmpFOpConversion,
                        ShiftLeftConversionPattern, OrIConversionPattern, SelectConversionPattern,
                        LoadOpConversion, StoreOpConversion,
                        /*MoveConstantUp, */CallOpConversion>(converter, &getContext());

                if (failed(applyPartialConversion(designOp, target, std::move(patterns)))) {
                    llvm::errs() << "conversion fail" << "\n";
                    signalPassFailure();
                }
            }
            {
                ConversionTarget target(getContext());
                RewritePatternSet patterns(&getContext());

                target.addLegalDialect<tor::TORDialect>();
                target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp op) {
                    if (!llvm::isa<tor::DesignOp>(op->getParentOp()))
                        return false;
                    return true;
                });

                patterns.add<MoveConstantUp>(&getContext());

                if (failed(applyPartialConversion(designOp, target, std::move(patterns))))
                    signalPassFailure();
            }

            // {
            //     designOp.walk([&](tor::FuncOp op) {
            //         RewritePatternSet rPatterns(&getContext());
            //         rPatterns.insert<DesignOpPattern>(&getContext());
            //         if (failed(applyOpPatternsAndFold(op, std::move(rPatterns))))
            //             WalkResult::interrupt();
            //         WalkResult::advance();
            //     });
            // }

        }
    };

} // namespace

namespace mlir {

    std::unique_ptr<OperationPass<tor::DesignOp>> createSCFToTORPass() {
        return std::make_unique<ConvertInputPass>();
    }

} // namespace mlir
