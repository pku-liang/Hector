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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"

#include "TOR/TORDialect.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#include <set>
#include <iostream>
#include <fstream>
#include <filesystem>

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

    inline void saveOldAttrWithName(mlir::Operation *newOp,
                                    mlir::Operation *op, std::string type) {
        if (auto attr = op->getAttr(type)) {
            newOp->setAttr(type, attr);
        }
    }

    inline void saveOldArgAttrWithName(mlir::Operation *newOp,
                                       mlir::Operation *op, std::string type,
                                       unsigned oldIndex) {
        if (auto attr = op->getAttr(type + "_" + llvm::Twine(oldIndex).str())) {
            newOp->setAttr(type, attr);
        }
    }

    inline void saveOldArgLineAttrWithName(mlir::Operation *newOp,
                                           mlir::Operation *op, std::string type,
                                           unsigned oldIndex) {
        if (auto attr = op->getAttr(type + "_arg_" + llvm::Twine(oldIndex).str() + "-line")) {
            newOp->setAttr(type + "-line", attr);
        }
    }

    struct FlattenArray : public OpRewritePattern<ModuleOp> {
        using OpRewritePattern<ModuleOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ModuleOp module, PatternRewriter &rewriter) const override {
            module.walk([&](FuncOp op) {
                rewriter.eraseOp(op);
            });

            for (auto &sop: *(module.getBody())) {
                if (auto design = dyn_cast<tor::DesignOp>(sop)) {
                    if (design->getAttr("flatten-array"))
                        return failure();
                    design->setAttr("flatten-array", IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
                    for (auto &op: (design.getBody().front())) {
                        if (auto alloc = dyn_cast<tor::AllocOp>(op)) {
                            auto memref = alloc.getMemref();
                            bool R = false;
                            bool W = false;
                            std::string RW;
                            for (auto &use: memref.getUses()) {
                                if (isa<tor::LoadOp>(use.getOwner())) {
                                    R = true;
                                } else if (isa<tor::StoreOp>(use.getOwner())) {
                                    W = true;
                                } else {
                                    assert(false);
                                }
                            }
                            if (R) RW += "r";
                            if (W) RW += "w";
                            auto shape = memref.getType().getShape();
                            int64_t size = 1;
                            SmallVector<int64_t> one;
                            for (auto s: shape) {
                                size *= s;
                            }
                            one.push_back(size);
                            for (auto &use: memref.getUses()) {
                                if (auto load = dyn_cast<tor::LoadOp>(use.getOwner())) {
                                    auto lastInsertion = rewriter.saveInsertionPoint();
                                    rewriter.setInsertionPoint(load);
                                    if (load.getIndices().empty()) {
                                        auto index = rewriter.create<ConstantIndexOp>(load.getLoc(), 0);
                                        load->insertOperands(1, index.getResult());
                                    } else {
                                        auto last_idx = load.getIndices()[0];
                                        for (unsigned i = 1; i < shape.size(); ++i) {
                                            int length = shape[i];
                                            if (!(length & (length - 1))) {
                                                int width = log2(length);
                                                auto size = rewriter.create<ConstantIndexOp>(load.getLoc(), width);
                                                auto mul = rewriter.create<ShLIOp>(load.getLoc(), last_idx,
                                                                                   size.getResult());
                                                auto add = rewriter.create<AddIOp>(load.getLoc(), load.getIndices()[i],
                                                                                   mul.getResult());
                                                last_idx = add.getResult();
                                            } else {
                                                auto size = rewriter.create<ConstantIndexOp>(load.getLoc(), length);
                                                auto mul = rewriter.create<MulIOp>(load.getLoc(), last_idx,
                                                                                   size.getResult());
                                                auto add = rewriter.create<AddIOp>(load.getLoc(), load.getIndices()[i],
                                                                                   mul.getResult());
                                                last_idx = add.getResult();
                                            }
                                        }
                                        load->setOperand(1, last_idx);
                                        for (unsigned i = shape.size() - 1; i >= 1; --i) {
                                            load->eraseOperand(i + 1);
                                        }
                                    }
                                    rewriter.restoreInsertionPoint(lastInsertion);
                                } else if (auto store = dyn_cast<tor::StoreOp>(use.getOwner())) {
                                    auto lastInsertion = rewriter.saveInsertionPoint();
                                    rewriter.setInsertionPoint(store);
                                    if (store.getIndices().empty()) {
                                        auto index = rewriter.create<ConstantIndexOp>(store.getLoc(), 0);
                                        store->insertOperands(2, index.getResult());
                                    } else {
                                        auto last_idx = store.getIndices()[0];
                                        for (unsigned i = 1; i < shape.size(); ++i) {
                                            int length = shape[i];
                                            if (!(length & (length - 1))) {
                                                int width = log2(length);
                                                auto size = rewriter.create<ConstantIndexOp>(store.getLoc(), width);
                                                auto mul = rewriter.create<ShLIOp>(store.getLoc(), last_idx,
                                                                                   size.getResult());
                                                // auto add = rewriter.create<OrIOp>(store.getLoc(), store.getIndices()[i], mul.getResult());
                                                auto add = rewriter.create<AddIOp>(store.getLoc(),
                                                                                   store.getIndices()[i],
                                                                                   mul.getResult());
                                                last_idx = add.getResult();
                                            } else {
                                                auto size = rewriter.create<ConstantIndexOp>(store.getLoc(), length);
                                                auto mul = rewriter.create<MulIOp>(store.getLoc(), last_idx,
                                                                                   size.getResult());
                                                auto add = rewriter.create<AddIOp>(store.getLoc(),
                                                                                   store.getIndices()[i],
                                                                                   mul.getResult());
                                                last_idx = add.getResult();
                                            }
                                        }
                                        store->setOperand(2, last_idx);
                                        for (unsigned i = shape.size() - 1; i >= 1; --i) {
                                            store->eraseOperand(i + 2);
                                        }
                                    }
                                    rewriter.restoreInsertionPoint(lastInsertion);
                                }
                            }
                            auto newType = tor::MemRefType::get(one, memref.getType().getElementType(), {},
                                                                StringAttr::get(getContext(), RW));
                            alloc.getResult().setType(newType);
                        }
                    }
                }
            }
            return success();
        }
    };
    
    struct MoveWhileOp : public OpRewritePattern<mlir::ModuleOp> {
        using OpRewritePattern<mlir::ModuleOp>::OpRewritePattern;

        LogicalResult
        matchAndRewrite(mlir::ModuleOp op, PatternRewriter &rewriter) const override {
            auto design = llvm::dyn_cast<tor::DesignOp>(op.getBody()->front());
            if (design->getAttr("move-while")) return failure();
            design->setAttr("move-while", IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
            SmallVector<scf::WhileOp> ops;
            design.walk([&](scf::WhileOp op) { ops.push_back(op); });
            for (auto &op: ops) {
                if (isa<scf::ConditionOp>(op.getBefore().front().begin())) continue;
                rewriter.setInsertionPoint(op);
                auto tmp_op = cast<scf::WhileOp>(rewriter.clone(*op));
                rewriter.setInsertionPointAfter(op);
                auto tmp_op_2 = cast<scf::WhileOp>(rewriter.clone(*op));
                for (unsigned idx = 0; idx < tmp_op.getBefore().getNumArguments(); ++idx) {
                    auto arg = tmp_op.getBefore().getArgument(idx);
                    SmallVector<std::pair<Operation *, int>> pairs;
                    for (auto &use: arg.getUses()) {
                        pairs.push_back(std::make_pair(use.getOwner(), use.getOperandNumber()));
                    }
                    for (auto pair: pairs) {
                        pair.first->setOperand(pair.second, tmp_op.getInits()[idx]);
                    }
                }

                rewriter.setInsertionPoint(tmp_op);
                for (auto &sop: tmp_op.getBefore().front()) {
                    if (isa<scf::ConditionOp>(sop)) continue;
                    sop.getResults().replaceAllUsesWith(rewriter.clone(sop)->getResults());
                }

                auto cond = cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
                unsigned idx = op.getNumOperands();
                op->insertOperands(idx, cond.getCondition());
                Block *new_block = new Block();
                op.getBefore().push_back(new_block);
                SmallVector<Location> locations(cond->getNumOperands(), op.getLoc());
                new_block->addArguments(cond->getOperandTypes(), locations);
                SmallVector<Value> values;
                for (unsigned idx = 1; idx < new_block->getNumArguments(); ++idx) {
                    values.push_back(new_block->getArgument(idx));
                }
                rewriter.setInsertionPointToStart(new_block);
                rewriter.create<scf::ConditionOp>(op.getLoc(), new_block->getArgument(0), values);
                op->setOperands(tmp_op.getBefore().front().getTerminator()->getOperands());
                op.getBefore().front().erase();
                rewriter.eraseOp(tmp_op);

                rewriter.setInsertionPoint(op);
                tmp_op = tmp_op_2;
                auto yield = cast<scf::YieldOp>(op.getAfter().front().getTerminator());
                for (unsigned idx = 0; idx < tmp_op.getBefore().getNumArguments(); ++idx) {
                    auto arg = tmp_op.getBefore().getArgument(idx);
                    SmallVector<std::pair<Operation *, int>> pairs;
                    for (auto &use: arg.getUses()) {
                        pairs.push_back(std::make_pair(use.getOwner(), use.getOperandNumber()));
                    }
                    for (auto pair: pairs) {
                        pair.first->setOperand(pair.second, yield.getOperand(idx));
                    }
                }
                rewriter.setInsertionPointToEnd(&op.getAfter().front());
                for (auto &sop: tmp_op.getBefore().front()) {
                    if (isa<scf::ConditionOp>(sop)) continue;
                    sop.getResults().replaceAllUsesWith(rewriter.clone(sop)->getResults());
                }
                rewriter.create<scf::YieldOp>(op.getLoc(), tmp_op.getBefore().front().getTerminator()->getOperands());
                rewriter.eraseOp(yield);
                rewriter.eraseOp(tmp_op);
            }
            return success();
        }
    };

    struct DesignOpPattern : OpRewritePattern<mlir::ModuleOp> {
        DesignOpPattern(MLIRContext *ctx, std::string top_function)
                : OpRewritePattern<mlir::ModuleOp>(ctx), top_function(top_function) {}

        LogicalResult matchAndRewrite(mlir::ModuleOp op, PatternRewriter &rewriter) const override {
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

    void solve(FuncOp op, std::string top_function, std::string resource, double clock, PatternRewriter &rewriter) {
        rewriter.setInsertionPoint(op);
        StringRef function_name = op.getName() == top_function ? "main" : op.getName();
        if (function_name != "main") {
            for (auto arg: op.getArgumentTypes()) {
                if (isa<MemRefType, tor::MemRefType>(arg)) {
                    return;
                }
            }
        }

        auto funcOp = rewriter.create<tor::FuncOp>(op.getLoc(), function_name,
                                                   op.getFunctionType());
        funcOp->setAttr("clock", FloatAttr::get(FloatType::getF32(op.getContext()), clock));
        funcOp->setAttr("resource", StringAttr::get(op.getContext(), resource));
        if (auto IIAttr = op->getAttr("II")) {
            funcOp->setAttr("II", IIAttr);
            funcOp->setAttr("pipeline", StringAttr::get(op.getContext(), "func"));
        }
        saveOldAttrWithName(funcOp, op, "dataflow");
        saveOldAttrWithName(funcOp, op, "dataflow-line");
        rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());
        rewriter.setInsertionPointToStart(op->getBlock());
        funcOp.walk([&](LLVM::UndefOp op) {
            auto lastInsertion = rewriter.saveInsertionPoint();
            rewriter.setInsertionPoint(op);
            Value undef;
            if (op.getRes().getType().isIntOrIndex()) {
                undef = rewriter.create<ConstantIntOp>(op.getLoc(), 0,
                                                       op.getRes().getType().getIntOrFloatBitWidth()).getResult();
            } else if (op.getRes().getType().isF32()) {
                undef = rewriter.create<ConstantFloatOp>(op.getLoc(), APFloat(0.0f),
                                                         rewriter.getF32Type()).getResult();
            } else if (op.getRes().getType().isF64()) {
                undef = rewriter.create<ConstantFloatOp>(op.getLoc(), APFloat(0.0),
                                                         rewriter.getF64Type()).getResult();
            } else {
                assert(false);
            }
            op.getRes().replaceAllUsesWith(undef);
            rewriter.restoreInsertionPoint(lastInsertion);
            rewriter.eraseOp(op);
        });
        SmallVector<Operation *, 8> opsToErase;

        for (auto *op : opsToErase)
            rewriter.eraseOp(op);
        funcOp.walk([&](ReturnOp op) {
            rewriter.setInsertionPoint(op);
            rewriter.create<tor::ReturnOp>(op.getLoc(), op.getOperands());
            rewriter.eraseOp(op);
        });
        rewriter.setInsertionPointToStart(op->getBlock());
        funcOp.walk([&](memref::AllocaOp alloca) {    // 临时变量
            auto memref = alloca.getMemref().getType();
            tor::AllocOp alloc;
            if (memref.getShape().empty()) {
                auto newType = tor::MemRefType::get({1}, memref.getElementType(), {},
                                                    StringAttr::get(op.getContext(), ""));
                alloc = rewriter.create<tor::AllocOp>(op.getLoc(), newType);
                alloca.getResult().replaceAllUsesWith(alloc.getResult());
                // alloc.setLocalTypeAttr(StringAttr::get(alloc.getContext(), "local"));
            } else {
                auto newType = tor::MemRefType::get(memref.getShape(), memref.getElementType(), {},
                                                    StringAttr::get(op.getContext(), ""));
                alloc = rewriter.create<tor::AllocOp>(op.getLoc(), newType);
                alloca.getResult().replaceAllUsesWith(alloc.getResult());
                // alloc.setLocalTypeAttr(StringAttr::get(alloc.getContext(), "local"));
            }
            saveOldAttrWithName(alloc, alloca, "bind_storage_type");
            saveOldAttrWithName(alloc, alloca, "bind_storage-line");
            saveOldAttrWithName(alloc, alloca, "mode");
            saveOldAttrWithName(alloc, alloca, "bus");
            saveOldAttrWithName(alloc, alloca, "offset");
            saveOldAttrWithName(alloc, alloca, "ARLEN");
            saveOldAttrWithName(alloc, alloca, "AWLEN");
            saveOldAttrWithName(alloc, alloca, "max_widen_bitwidth");
            saveOldAttrWithName(alloc, alloca, "initial_addr");
            saveOldAttrWithName(alloc, alloca, "interface-storage_type");
            saveOldAttrWithName(alloc, alloca, "num_read_outstanding");
            saveOldAttrWithName(alloc, alloca, "num_write_outstanding");
            rewriter.eraseOp(alloca);
        });
        funcOp.walk([&](memref::AllocOp alloca) {
            auto memref = alloca.getMemref().getType();
            if (memref.getShape().empty()) {
                auto newType = tor::MemRefType::get({1}, memref.getElementType(), {},
                                                    StringAttr::get(op.getContext(), ""));
                auto alloc = rewriter.create<tor::AllocOp>(op.getLoc(), newType);
                alloca.getResult().replaceAllUsesWith(alloc.getResult());
            } else {
                auto newType = tor::MemRefType::get(memref.getShape(), memref.getElementType(), {},
                                                    StringAttr::get(op.getContext(), ""));
                auto alloc = rewriter.create<tor::AllocOp>(op.getLoc(), newType);
                alloca.getResult().replaceAllUsesWith(alloc.getResult());
            }
            rewriter.eraseOp(alloca);
        });
        funcOp.walk([&](memref::LoadOp load) {
            rewriter.setInsertionPoint(load);
            auto new_load = rewriter.create<tor::LoadOp>(load.getLoc(), load.getResult().getType(),
                                                         load.getOperand(0), 0, 0,
                                                         load.getIndices());
            load.getResult().replaceAllUsesWith(new_load.getResult());
            if (load->getAttr("distance")) {
              new_load->setAttr("distance", load->getAttr("distance"));
            }
            rewriter.eraseOp(load);
        });
        funcOp.walk([&](memref::StoreOp store) {
            rewriter.setInsertionPoint(store);
            auto new_store = rewriter.create<tor::StoreOp>(store.getLoc(), store.getValue(),
                                                           store.getOperand(1), 0, 0,
                                                           store.getIndices());
            if (store->getAttr("distance")) {
              new_store->setAttr("distance", store->getAttr("distance"));
            }
            rewriter.eraseOp(store);
        });
        funcOp.walk([&](memref::CastOp cast) {
            cast.getResult().replaceAllUsesWith(cast.getOperand());
            rewriter.eraseOp(cast);
        });
        rewriter.setInsertionPointToStart(op->getBlock());
        unsigned idx = 0, oldIdx = 0;
        while (idx < funcOp.getNumArguments()) {
          auto arg = funcOp.getArgument(idx);
          if (auto memref = dyn_cast<MemRefType>(arg.getType())) {
            auto newType =
                tor::MemRefType::get(memref.getShape(), memref.getElementType(),
                                     {}, StringAttr::get(op.getContext(), ""));
            auto alloc = rewriter.create<tor::AllocOp>(
                funcOp.getLoc(), newType);
            saveOldArgAttrWithName(alloc, op, "bind_storage_type", oldIdx);
            saveOldArgLineAttrWithName(alloc, op, "bind_storage", oldIdx);
            saveOldArgLineAttrWithName(alloc, op, "interface", oldIdx);
            saveOldArgAttrWithName(alloc, op, "mode", oldIdx);
            saveOldArgAttrWithName(alloc, op, "bus", oldIdx);
            saveOldArgAttrWithName(alloc, op, "offset", oldIdx);
            saveOldArgAttrWithName(alloc, op, "ARLEN", oldIdx);
            saveOldArgAttrWithName(alloc, op, "AWLEN", oldIdx);
            saveOldArgAttrWithName(alloc, op, "max_widen_bitwidth", oldIdx);
            saveOldArgAttrWithName(alloc, op, "initial_addr", oldIdx);
            saveOldArgAttrWithName(alloc, op, "interface-storage_type", oldIdx);
            saveOldArgAttrWithName(alloc, op, "num_read_outstanding", oldIdx);
            saveOldArgAttrWithName(alloc, op, "num_write_outstanding", oldIdx);
            arg.replaceAllUsesWith(alloc.getResult());
            funcOp.eraseArgument(idx);
          } else {
            idx += 1;
          }
          oldIdx += 1;
        }
        static int func_num = 0;
        funcOp.walk([&](CallOp call) {
            // bool hasMem = false;
            // for (auto arg_type: call.getCalleeType().getInputs()) {
            //     if (isa<tor::MemRefType, MemRefType, tor::StreamType>(arg_type)) {
            //         hasMem = true;
            //         break;
            //     }
            // }
            bool hasMem = true;
            if (hasMem) {
                auto design = dyn_cast<tor::DesignOp>(funcOp->getParentOp());
                for (auto func: design.getOps<func::FuncOp>()) {
                    if (func.getSymName() == call.getCallee()) {
                        rewriter.setInsertionPoint(funcOp);
                        auto new_func = cast < func::FuncOp > (rewriter.clone(*func));
                        new_func.setSymName(new_func.getSymName().str() + "_" + std::to_string(++func_num));
                        SmallVector<Value, 8> new_operands;
                        unsigned idx = 0;
                        for (auto &operand: call->getOpOperands()) {
                            if (isa<tor::MemRefType, MemRefType>(operand.get().getType())) {
                                new_func.getArgument(idx).replaceAllUsesWith(operand.get());
                                new_func.eraseArgument(idx);
                            } else {
                                new_operands.push_back(operand.get());
                                idx += 1;
                            }
                        }
                        rewriter.setInsertionPoint(call);
                        rewriter.replaceOpWithNewOp<tor::CallOp>(call, call.getResultTypes(),
                                                                 new_func.getSymName(), 0, 0, new_operands);
                        solve(new_func, top_function, resource, clock, rewriter);
                        return;
                    }
                }
            } else {
                rewriter.setInsertionPoint(call);
                auto newOp = rewriter.create<tor::CallOp>(call.getLoc(), call.getResultTypes(),
                                                          call.getCallee(), 0, 0, call->getOperands());
                rewriter.replaceOp(call, newOp.getResults());
                auto design = dyn_cast<tor::DesignOp>(funcOp->getParentOp());
                for (auto func: design.getOps<func::FuncOp>()) {
                    if (func.getSymName() == call.getCallee()) {
                        solve(func, top_function, resource, clock, rewriter);
                        return;
                    }
                }
            }
        });

        rewriter.eraseOp(op);
    }

    struct FuncCallPattern : public OpRewritePattern<FuncOp> {
        FuncCallPattern(MLIRContext *ctx, std::string top_function, std::string resource, double clock)
                : OpRewritePattern<FuncOp>(ctx), top_function(top_function), resource(resource), clock(clock) {}

        LogicalResult matchAndRewrite(FuncOp op, PatternRewriter &rewriter) const override {
            std::map<std::string, Value> memory_mapping;
            auto design = op->getParentOp();
            design->walk([&](memref::GlobalOp global) {
                rewriter.setInsertionPoint(global);
                auto memref = global.getType();
                tor::AllocOp alloc;
                if (memref.getShape().empty()) {
                    auto newType = tor::MemRefType::get({1}, memref.getElementType(), {},
                                                        StringAttr::get(global.getContext(), ""));
                    auto value = global.getInitialValue();
                    auto value_d = value.has_value() && !value.value().isa<mlir::UnitAttr>() ? value.value() : nullptr;
                    alloc = rewriter.create<tor::AllocOp>(global.getLoc(), newType);
                    memory_mapping[global.getSymName().str()] = alloc.getResult();
                } else {
                    auto newType = tor::MemRefType::get(memref.getShape(), memref.getElementType(), {},
                                                        StringAttr::get(global.getContext(), ""));
                    auto value = global.getInitialValue();
                    auto value_d = value.has_value() && !value.value().isa<mlir::UnitAttr>() ? value.value() : nullptr;
                    alloc = rewriter.create<tor::AllocOp>(global.getLoc(), newType);
                    memory_mapping[global.getSymName().str()] = alloc.getResult();
                }
                saveOldAttrWithName(alloc, global, "bind_storage_type");
                saveOldAttrWithName(alloc, global, "bind_storage-line");
                saveOldAttrWithName(alloc, global, "interface-line");
                saveOldAttrWithName(alloc, global, "mode");
                saveOldAttrWithName(alloc, global, "bus");
                saveOldAttrWithName(alloc, global, "offset");
                saveOldAttrWithName(alloc, global, "ARLEN");
                saveOldAttrWithName(alloc, global, "AWLEN");
                saveOldAttrWithName(alloc, global, "max_widen_bitwidth");
                saveOldAttrWithName(alloc, global, "initial_addr");
                saveOldAttrWithName(alloc, global, "interface-storage_type");
                saveOldAttrWithName(alloc, global, "num_read_outstanding");
                saveOldAttrWithName(alloc, global, "num_write_outstanding");
                rewriter.eraseOp(global);
            });
            design->walk([&](memref::GetGlobalOp get_global) {
                get_global.getResult().replaceAllUsesWith(memory_mapping[get_global.getName().str()]);
                rewriter.eraseOp(get_global);
            });
            solve(op, top_function, resource, clock, rewriter);
            return success();
        }

        std::string top_function, resource;
        double clock;
    };


    struct MulIOpConversion : public OpRewritePattern<func::FuncOp> {
        using OpRewritePattern<func::FuncOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(func::FuncOp funcOp, PatternRewriter &rewriter) const override {
            if (funcOp->getAttr("mul-convert"))
                return failure();
            SmallVector<std::pair<MulIOp, ShLIOp>> replace;
            funcOp.walk([&](MulIOp op) {
                auto val = op.getRhs();
                APInt int_val;
                if (matchPattern(val, m_ConstantInt(&int_val))) {
                    if (int_val.isPowerOf2()) {
                        int width = int_val.logBase2();
                        rewriter.setInsertionPoint(op);
                        if (op.getRhs().getType().isIndex()) {
                            auto constant = rewriter.create<ConstantIndexOp>(op.getLoc(), width);
                            auto shift_left = rewriter.create<ShLIOp>(op.getLoc(), op.getLhs(), constant.getResult());
                            replace.push_back(std::make_pair(op, shift_left));
                        } else {
                            auto constant = rewriter.create<ConstantIntOp>(op.getLoc(), width, op.getRhs().getType());
                            auto shift_left = rewriter.create<ShLIOp>(op.getLoc(), op.getLhs(), constant.getResult());
                            replace.push_back(std::make_pair(op, shift_left));
                        }
                    }
                }
            });
            for (auto &pair: replace) {
                pair.first.getResult().replaceAllUsesWith(pair.second.getResult());
            }
            funcOp->setAttr("mul-convert",
                            IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
            return success();
        }
    };

    struct MulIOpErase : public OpConversionPattern<MulIOp> {
        using OpConversionPattern<MulIOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(MulIOp op, MulIOp::Adaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override {
            if (!op.use_empty())
                return failure();
            rewriter.eraseOp(op);
            return success();
        }
    };

    struct ConvertInputPass : ConvertInputBase<ConvertInputPass> {
        void runOnOperation() override {
            auto moduleOp = getOperation();
            // Huang Ruibo comments on this line of code to preserve the 
            // pragma report Attr.
            // moduleOp->setAttrs(DictionaryAttr::getWithSorted(&getContext(), {}));
            {
                moduleOp.walk([&](func::FuncOp op) {
                    RewritePatternSet Patterns(op.getContext());
                    Patterns.add<MulIOpConversion>(op.getContext());
                    if (failed(applyOpPatternsAndFold(op.getOperation(),std::move(Patterns))))
                        signalPassFailure();
                });
            }
            {
                ConversionTarget target(getContext());
                RewritePatternSet patterns(&getContext());
                target.addDynamicallyLegalOp<MulIOp>([](MulIOp op) {
                    if (op.use_empty())
                        return false;
                    return true;
                });
                patterns.add<MulIOpErase>(&getContext());
                if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
                    signalPassFailure();
            }

            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<DesignOpPattern>(moduleOp.getContext(), top_function);

                if (failed(applyOpPatternsAndFold(moduleOp.getOperation(), std::move(Patterns))))
                    signalPassFailure();
            }
            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<MoveWhileOp>(&getContext());
                if (failed(applyOpPatternsAndFold(moduleOp.getOperation(), std::move(Patterns))));
            }

            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<FuncCallPattern>(&getContext(), top_function, resource, clock);

                moduleOp->walk([&](func::FuncOp op) {
                    if (op.getSymName() == top_function) {
                        if (failed(applyOpPatternsAndFold(op.getOperation(), std::move(Patterns))))
                            signalPassFailure();
                    }
                });
            }

            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<FlattenArray>(&getContext());

                if (failed(applyOpPatternsAndFold(moduleOp.getOperation(),std::move(Patterns)))) {
                      signalPassFailure();
                  }
            }
        }
    };

} // namespace

namespace mlir {

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertInputPass() {
        return std::make_unique<ConvertInputPass>();
    }

} // namespace mlir
