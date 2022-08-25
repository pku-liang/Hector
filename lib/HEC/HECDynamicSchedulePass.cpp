#include "HEC/PassDetail.h"
#include "mlir/Analysis/Utils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "TOR/TOR.h"
#include "TOR/TORDialect.h"
#include "HEC/HEC.h"
#include "HEC/HECDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include <mlir/Transforms/DialectConversion.h>
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Analysis/Liveness.h"

#include <map>
#include <set>
#include <queue>
#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>

#define DEBUG_TYPE "dynamic-schedule"

namespace mlir {
    namespace dynamic {
#define TIME_NODE 105

        struct TimeEdge {
            int from;
            int to;
            bool loop_exit;
            Attribute attr;
            std::vector<Operation *> ops;

            TimeEdge(int _from, int _to, Attribute _attr, bool _exit = false) {
                from = _from;
                to = _to;
                attr = _attr;
                loop_exit = _exit;
                ops.clear();
            }
        };

        std::vector<TimeEdge> timeGraph[TIME_NODE];
        int ifEnd[TIME_NODE], ifBegin[TIME_NODE];
        Operation *ifOp[TIME_NODE];
        int whileEnd[TIME_NODE], whileBegin[TIME_NODE];
        Operation *whileOp[TIME_NODE];
        int forEnd[TIME_NODE], forBegin[TIME_NODE];
        Operation *forOp[TIME_NODE];
        Operation *succOp[TIME_NODE];
        std::map<std::string, int> instanceCounts;

        void bind_operation(int src, int dest, Operation *op) {
            for (auto &edge : timeGraph[src]) {
                if (edge.to == dest) {
                    edge.ops.push_back(op);
                }
            }
        }

        Value control_signal;

        struct Liveness {
            Value val;

            Liveness(const Value &_val) {
                val = _val;
            }

            Liveness(const Liveness &live) {
                val = live.val;
            }

            bool operator<(const Liveness &x) const {
                if (val == x.val) {
                    return false;
                }
                return val.getImpl() < x.val.getImpl();

            }

            bool operator==(const Liveness &x) const {
                return val == x.val;
            }

            operator bool() const {
                return val != control_signal;
            }

            void dump() {
                if (!val) {
                    std::cerr << "Control Signal!\n";
                    return;
                }
                val.dump();
            }
        };

        std::map<Operation *, std::set<Liveness>> liveins, liveouts;

        bool update_liveout(Operation *op, const std::set<Liveness> &set) {
            if (!op)
                return false;
            auto &livein = liveins[op];
            auto &liveout = liveouts[op];
            auto in_count = livein.size(), out_count = liveout.size();
            for (auto val : set) {
                if (val.val.getDefiningOp() == op) {
                    if (auto for_op = dyn_cast<tor::ForOp>(op)) {
                        for (auto for_val : for_op.getBody()->getTerminator()->getOperands()) {
                            liveout.insert(for_val);
                        }
                    }
                    continue;
                }
                liveout.insert(val.val);
            }
            return (in_count != livein.size() || out_count != liveout.size());
        }

        void livenessAnalysis(Operation *op) {
#define LIVE_INSERT(VAL) if(VAL.isa<BlockArgument>()||(!isa<tor::AllocOp>(VAL.getDefiningOp())))livein.insert(VAL);
            if (!op || isa<ConstantOp>(op))
                return;
            auto &livein = liveins[op];
            livein.insert(control_signal);
            auto &liveout = liveouts[op];
            auto in_count = livein.size(), out_count = liveout.size();
            if (auto forOp = dyn_cast<tor::ForOp>(op)) {
                std::set<Operation *> collect_ops;
                collect_ops.clear();
                for (auto &op : forOp.getRegion().front()) {
                    collect_ops.insert(&op);
                }
                bool flag = false;
                for (auto val : forOp.getRegion().front().getTerminator()->getOperands()) {
                    liveout.insert(val);
                    if (update_liveout(val.getDefiningOp(), liveout)) {
                        flag = true;
                    }
                }
                for (auto &op : forOp.getRegion().front()) {
                    if (isa<tor::TimeGraphOp, ConstantOp, tor::ReturnOp, tor::YieldOp>(op)) {
                        continue;
                    }
                    livenessAnalysis(&op);
                    for (auto &val : liveins[&op]) {
                        if (!val) {
                            continue;
                        }
                        if (val.val.isa<BlockArgument>()) {
                            auto arg = val.val.cast<BlockArgument>();
                            if (arg.getOwner()->getParentOp() != forOp) {
                                livein.insert(val);
                            }
                            continue;
                        }
                        if (!val.val.isa<BlockArgument>() && isa<ConstantOp>(val.val.getDefiningOp())) {
                            continue;
                        }
                        if (collect_ops.find(val.val.getDefiningOp()) == collect_ops.end()) {
                            LIVE_INSERT(val.val);
                        }
                    }
                }
                LIVE_INSERT(forOp.lowerBound());
                LIVE_INSERT(forOp.upperBound());
                LIVE_INSERT(forOp.step());
                for (auto init : forOp.initArgs()) {
                    LIVE_INSERT(init);
                }

                if (flag || in_count != livein.size() || out_count != liveout.size()) {
                    for (auto val : livein) {
                        if (!val) {
                            continue;
                        }
                        livenessAnalysis(val.val.getDefiningOp());
                    }
                    livenessAnalysis(op);
                }
            } else if (auto ifOp = dyn_cast<tor::IfOp>(op)) {
                std::set<Operation *> collect_ops;
                collect_ops.clear();
                std::cerr << ifOp->getNumRegions();
                for (unsigned idx = 0; idx < ifOp->getNumRegions(); ++idx) {
                    if (ifOp.getRegion(idx).empty()) {
                        continue;
                    }
                    for (auto &op : ifOp.getRegion(idx).front()) {
                        collect_ops.insert(&op);
                    }
                }
                LIVE_INSERT(ifOp.condition());
                bool flag = false;
                for (unsigned idx = 0; idx < ifOp->getNumRegions(); ++idx) {
                    if (ifOp.getRegion(idx).empty()) {
                        continue;
                    }
                    for (auto val : ifOp.getRegion(idx).front().getTerminator()->getOperands()) {
                        liveout.insert(val);
                        if (update_liveout(val.getDefiningOp(), liveout)) {
                            flag = true;
                        }
                    }
                }
                for (unsigned idx = 0; idx < ifOp->getNumRegions(); ++idx) {
                    if (ifOp.getRegion(idx).empty()) {
                        continue;
                    }
                    for (auto &op : ifOp.getRegion(idx).front()) {
                        if (isa<tor::TimeGraphOp, ConstantOp>(op)) {
                            continue;
                        }
                        livenessAnalysis(&op);
                        for (auto &val : liveins[&op]) {
                            if (!val) {
                                continue;
                            }
                            if (val.val.isa<BlockArgument>()) {
                                auto arg = val.val.cast<BlockArgument>();
                                if (arg.getOwner()->getParentOp() != ifOp) {
                                    livein.insert(val);
                                }
                                continue;
                            }
                            if (!val.val.isa<BlockArgument>() && isa<ConstantOp>(val.val.getDefiningOp())) {
                                continue;
                            }
                            if (collect_ops.find(val.val.getDefiningOp()) == collect_ops.end()) {
                                LIVE_INSERT(val.val);
                            }
                        }
                    }
                }
                if (flag || in_count != livein.size() || out_count != liveout.size()) {
                    for (auto val : livein) {
                        if (!val) {
                            continue;
                        }
                        livenessAnalysis(val.val.getDefiningOp());
                    }
                    livenessAnalysis(op);
                }
            } else if (auto whileOp = dyn_cast<tor::WhileOp>(op)) {
                assert(false);
            } else if (auto callOp = dyn_cast<tor::CallOp>(op)) {
                for (auto val : callOp.getArgOperands()) {
                    LIVE_INSERT(val);
                }
            } else {
                for (auto val : op->getOperands()) {
                    LIVE_INSERT(val);
                }
            }
#undef LIVE_INSERT
        }

        void livenessAnalysis(tor::FuncOp funcOp) {
            auto &livein = liveins[funcOp];
            auto &liveout = liveouts[funcOp];
            auto in_count = livein.size(), out_count = liveout.size();
            std::set<Operation *> collect_ops;
            collect_ops.clear();
            for (auto &op : funcOp.getRegion().front()) {
                collect_ops.insert(&op);
            }
//            for (auto &op : funcOp.getRegion().front()) {
//                for (auto val : op.getOperands()) {
//                    if (val.isa<BlockArgument>()) {
//                        auto arg = val.cast<BlockArgument>();
//                        if (arg.getOwner()->getParentOp() != funcOp) {
//                            livein.insert(val);
//                        }
//                        continue;
//                    }
//                }
//            }
            bool flag = false;
            for (auto val : funcOp.getRegion().front().getTerminator()->getOperands()) {
                liveout.insert(val);
                if (update_liveout(val.getDefiningOp(), liveout)) {
                    flag = true;
                }
            }
            for (auto &op : funcOp.getRegion().front()) {
                if (isa<tor::TimeGraphOp, ConstantOp>(op)) {
                    continue;
                }
                livenessAnalysis(&op);
                for (auto &val : liveins[&op]) {
                    if (!val) {
                        continue;
                    }
                    if (val.val.isa<BlockArgument>()) {
                        auto arg = val.val.cast<BlockArgument>();
                        if (arg.getOwner()->getParentOp() != funcOp) {
                            livein.insert(val);
                        }
                        continue;
                    }
                    if (collect_ops.find(val.val.getDefiningOp()) == collect_ops.end()) {
                        livein.insert(val);
                    }
                }
            }
            if (flag || in_count != livein.size() || out_count != liveout.size()) {
                for (auto val : livein) {
                    if (!val) {
                        continue;
                    }
                    livenessAnalysis(val.val.getDefiningOp());
                }
                livenessAnalysis(funcOp);
            }
        }

        void analysis_top(tor::FuncOp funcOp) {
            livenessAnalysis(funcOp);
            return;
            std::cerr << "LIVE~IN~~~~~~~~~~~~~~~~\n";
            for (auto pair : liveins) {
                pair.first->dump();
                for (auto val : pair.second) {
                    if (!val) {
                        val.dump();
                        continue;
                    }
                    val.val.dump();
                    if (val.val.getDefiningOp()) {
                        val.val.getDefiningOp()->dump();
                    } else {
                        val.val.getParentBlock()->getParentOp()->dump();
                    }
                }
                std::cerr << "~~~~~~~~~~~~~~~~~~~~~~~~\n";
            }
            exit(-1);
            std::cerr << "LIVE-OUT----------------\n";
            for (auto pair : liveouts) {
                pair.first->dump();
                for (auto val : pair.second) {
                    val.val.dump();
                }
                std::cerr << "~~~~~~~~~~~~~~~~~~~~~~~~\n";
            }
            exit(-1);
        }

        std::map<std::pair<Operation *, Liveness>, Value> hec_operation;
        std::map<Operation *, Operation *> new_operation;
        int count = 0;

        hec::ComponentOp *TopComp;

        hec::PrimitiveOp
        create_primitive(Location loc, llvm::SmallVector<mlir::Type, 4> &types, std::string primitive, std::string name,
                         PatternRewriter &rewriter) {
            auto lastInsertion = rewriter.saveInsertionPoint();
            rewriter.setInsertionPoint(TopComp->getGraphOp());
            auto context = rewriter.getContext();
            auto instanceName = StringAttr::get(context, llvm::StringRef(name + std::to_string(count++)));
            auto primitiveName = StringAttr::get(context, primitive);
            auto primOp = rewriter.create<hec::PrimitiveOp>(loc, mlir::TypeRange(types), instanceName,
                                                            primitiveName);
            rewriter.restoreInsertionPoint(lastInsertion);
            return primOp;
        }

        Value get_value(Operation *op, Value val, PatternRewriter &rewriter) {
            if (val != control_signal && val.getDefiningOp()) {
//                std::cerr << "GET: ";
//                op->dump();
//                std::cerr << "VAL: ";
//                val.dump();
                auto sop = val.getDefiningOp();
                if (isa<ConstantOp>(sop)) {
//                    return new_operation[sop]->getResult(0);
//                    if (hec_operation.find(std::make_pair(op, Liveness(val))) == hec_operation.end())
                    if (hec_operation[std::make_pair(op, Liveness(val))] == Value()) {
//                        std::cerr << "NO FOUND\n";
                        llvm::SmallVector<mlir::Type, 4> types;
                        types.push_back(rewriter.getIntegerType(1000));
                        types.push_back(sop->getResult(0).getType());
                        types.push_back(sop->getResult(0).getType());
                        auto constant = create_primitive(op->getLoc(), types, "constant", "const_", rewriter);
                        hec_operation[std::make_pair(op, Liveness(val))] = constant.getResult(2);
                        rewriter.create<hec::AssignOp>(op->getLoc(), constant.getResult(0),
                                                       get_value(op->getParentOp(), control_signal, rewriter), Value());
                        rewriter.create<hec::AssignOp>(op->getLoc(), constant.getResult(1),
                                                       new_operation[sop]->getResult(0), Value());
//                        constant->dump();
                    } else {
//                        std::cerr << "FOUND\n";
//                        hec_operation[std::make_pair(op, Liveness(val))].dump();
//                        std::cerr << (hec_operation[std::make_pair(op, Liveness(val))] == Value());
//                        std::cerr << "FOUND\n";
//                        exit(-1);
                    }
                    return hec_operation[std::make_pair(op, Liveness(val))];
                }
            }
            return hec_operation[std::make_pair(op, Liveness(val))];
        }

        std::map<Operation *, std::vector<Operation *>> loadSet;
        std::map<Operation *, std::vector<Operation *>> storeSet;
        std::map<Operation *, std::vector<Operation *>> loadstoreSet;
        std::map<Operation *, Operation *> memSet;

        void Generate_operation(Operation *op, PatternRewriter &rewriter) {
            if (!op || isa<ConstantOp>(op))
                return;
            std::cerr << "Generate: ";
            op->dump();
            for (auto val : liveins[op]) {
                if (val && val.val.getDefiningOp()) {
                    if (isa<ConstantOp>(val.val.getDefiningOp())) {
                        continue;
                    }
                }
                hec_operation[std::make_pair(op, val)] = hec_operation[std::make_pair(op->getParentOp(), val)];
            }
            if (auto forOp = dyn_cast<tor::ForOp>(op)) {
                llvm::SmallVector<Operation *, 4> mux_set;
                mux_set.resize(forOp.getBody()->getNumArguments());
                llvm::SmallVector<Operation *, 4> branch_set;
                branch_set.resize(forOp.getBody()->getNumArguments());
                hec::PrimitiveOp compare;
                llvm::SmallVector<mlir::Type, 4> types;
                types.push_back(forOp.upperBound().getType());
                types.push_back(forOp.upperBound().getType());
                types.push_back(rewriter.getI1Type());
                compare = create_primitive(forOp.getLoc(), types, "cmp_integer_sle", "cmpi_sle_",
                                           rewriter);

                //control signal
                auto control_val = Liveness(control_signal);
                types.clear();
                for (int loop = 0; loop < 3; ++loop) {
                    types.push_back(rewriter.getIntegerType(1000));
                }
                types.push_back(rewriter.getI1Type());
                auto c_merge = create_primitive(forOp.getLoc(), types, "control_merge", "m_", rewriter);
                types.clear();
                for (int loop = 0; loop < 2; ++loop) {
                    types.push_back(rewriter.getIntegerType(1000));
                }
                auto buffer = create_primitive(forOp.getLoc(), types, "buffer", "buf_", rewriter);
                types.clear();
                types.push_back(rewriter.getI1Type());
                for (int loop = 0; loop < 3; ++loop) {
                    types.push_back(rewriter.getIntegerType(1000));
                }
                auto branch = create_primitive(forOp.getLoc(), types, "branch", "b_", rewriter);
                rewriter.create<hec::AssignOp>(forOp.getLoc(), branch.getResult(1), buffer.getResult(1), Value());
                rewriter.create<hec::AssignOp>(forOp.getLoc(), buffer.getResult(0), c_merge.getResult(2),
                                               Value());
                rewriter.create<hec::AssignOp>(forOp.getLoc(), branch.getResult(0), compare.getResult(2), Value());
                rewriter.create<hec::AssignOp>(forOp.getLoc(), c_merge.getResult(0),
                                               get_value(forOp, control_val.val, rewriter), Value());
                auto merge_condition = c_merge.getResult(3);
                hec_operation[std::make_pair(forOp, control_val)] = branch.getResult(2);

                for (auto arg : forOp.getBody()->getArguments()) {
                    if (arg.getArgNumber() == 0) {
                        continue;
                    }
                    llvm::SmallVector<mlir::Type, 4> types;
                    for (int loop = 0; loop < 3; ++loop) {
                        types.push_back(arg.getType());
                    }
                    types.push_back(rewriter.getI1Type());
                    auto mux = create_primitive(forOp.getLoc(), types, "mux_dynamic", "m_", rewriter);
                    mux_set[arg.getArgNumber()] = mux;
                    types.clear();
                    for (int loop = 0; loop < 2; ++loop) {
                        types.push_back(arg.getType());
                    }
                    auto buffer = create_primitive(forOp.getLoc(), types, "buffer", "buf_", rewriter);
                    types.clear();
                    types.push_back(rewriter.getI1Type());
                    for (int loop = 0; loop < 3; ++loop) {
                        types.push_back(arg.getType());
                    }
                    auto branch = create_primitive(forOp.getLoc(), types, "branch", "b_", rewriter);
                    branch_set[arg.getArgNumber()] = branch;
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), branch.getResult(1), buffer.getResult(1), Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), buffer.getResult(0), mux.getResult(2), Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), branch.getResult(0), compare.getResult(2), Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), mux.getResult(0),
                                                   get_value(forOp, forOp.initArgs()[arg.getArgNumber() - 1], rewriter),
                                                   Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), mux.getResult(3), merge_condition, Value());
                    hec_operation[std::make_pair(forOp, arg)] = branch.getResult(2);

                }
#define LIVE_INSERT(VAL) if(VAL.isa<BlockArgument>()||(!isa<tor::AllocOp>(VAL.getDefiningOp())))livein.insert(VAL);
                std::set<Liveness> livein;
//                livein.insert(control_signal);
                std::set<Operation *> collect_ops;
                collect_ops.clear();
                for (auto &op : forOp.getRegion().front()) {
                    collect_ops.insert(&op);
                }
                for (auto &op : forOp.getRegion().front()) {
                    if (isa<tor::TimeGraphOp, ConstantOp, tor::ReturnOp, tor::YieldOp>(op)) {
                        continue;
                    }
                    for (auto &val : liveins[&op]) {
                        if (!val) {
                            continue;
                        }
                        if (val.val.isa<BlockArgument>()) {
                            auto arg = val.val.cast<BlockArgument>();
                            if (arg.getOwner()->getParentOp() != forOp) {
                                livein.insert(val);
                            }
                            continue;
                        }
                        if (!val.val.isa<BlockArgument>() && isa<ConstantOp>(val.val.getDefiningOp())) {
                            continue;
                        }
                        if (collect_ops.find(val.val.getDefiningOp()) == collect_ops.end()) {
                            LIVE_INSERT(val.val);
                        }
                    }
                    LIVE_INSERT(forOp.upperBound());
                    LIVE_INSERT(forOp.step());
                }
#undef LIVE_INSERT
//                for (auto val : livein) {
//                    val.dump();
//                }
                auto lowerBound = get_value(forOp, forOp.lowerBound(), rewriter);
                auto upperBound = Value();
//                auto step = Value();
                for (auto val : liveins[forOp]) {
                    std::cerr << "-----------";
                    val.dump();
//                    if (!val) {
//                        continue;
//                    }
                    if (val && val.val.getDefiningOp() == forOp) {
                        continue;
                    }
                    if (livein.find(val) == livein.end()) {
                        std::cerr << "#######";
                        val.dump();
                        continue;
                    }
                    llvm::SmallVector<mlir::Type, 4> types;
                    for (int loop = 0; loop < 3; ++loop) {
                        types.push_back(val.val.getType());
                    }
                    types.push_back(rewriter.getI1Type());
                    auto mux = create_primitive(forOp.getLoc(), types, "mux_dynamic", "m_", rewriter);
                    types.clear();
                    for (int loop = 0; loop < 2; ++loop) {
                        types.push_back(val.val.getType());
                    }
                    auto buffer = create_primitive(forOp.getLoc(), types, "buffer", "buf_", rewriter);
                    types.clear();
                    types.push_back(rewriter.getI1Type());
                    for (int loop = 0; loop < 3; ++loop) {
                        types.push_back(val.val.getType());
                    }
                    auto branch = create_primitive(forOp.getLoc(), types, "branch", "b_", rewriter);
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), branch.getResult(1), buffer.getResult(1), Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), buffer.getResult(0), mux.getResult(2), Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), branch.getResult(0), compare.getResult(2), Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), mux.getResult(0),
                                                   get_value(forOp, val.val, rewriter), Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), mux.getResult(1), branch.getResult(2), Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), mux.getResult(3), merge_condition, Value());
                    hec_operation[std::make_pair(forOp, val)] = branch.getResult(2);
                    //FIXME: data with while and do_while
                    if (val.val == forOp.upperBound()) {
                        upperBound = buffer.getResult(1);
                    }
                }
                std::cerr << "Live in finish!\n";
                for (auto arg : forOp.getBody()->getArguments()) {
                    if (arg.getArgNumber() != 0) {
                        continue;
                    }
                    llvm::SmallVector<mlir::Type, 4> types;
                    for (int loop = 0; loop < 3; ++loop) {
                        types.push_back(arg.getType());
                    }
                    types.push_back(rewriter.getI1Type());
                    auto mux = create_primitive(forOp.getLoc(), types, "mux_dynamic", "m_", rewriter);
                    mux_set[arg.getArgNumber()] = mux;
                    types.clear();
                    for (int loop = 0; loop < 2; ++loop) {
                        types.push_back(arg.getType());
                    }
                    auto buffer = create_primitive(forOp.getLoc(), types, "buffer", "buf_", rewriter);
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), buffer.getResult(0),
                                                   mux.getResult(2), Value());
                    types.clear();
                    types.push_back(rewriter.getI1Type());
                    for (int loop = 0; loop < 3; ++loop) {
                        types.push_back(forOp.upperBound().getType());
                    }
                    auto branch = create_primitive(forOp.getLoc(), types, "branch", "b_", rewriter);
                    types.clear();
                    types.push_back(forOp.upperBound().getType());
                    types.push_back(branch.getResult(2).getType());
                    types.push_back(forOp.upperBound().getType());
                    auto add = create_primitive(forOp.getLoc(), types, "add_integer", "addi_", rewriter);
                    hec_operation[std::make_pair(forOp, arg)] = branch.getResult(2);

                    rewriter.create<hec::AssignOp>(forOp.getLoc(), mux.getResult(0), lowerBound, Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), compare.getResult(0), buffer.getResult(1),
                                                   Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), compare.getResult(1), upperBound, Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), branch.getResult(0), compare.getResult(2),
                                                   Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), branch.getResult(1), buffer.getResult(1),
                                                   Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), add.getResult(0), branch.getResult(2), Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), add.getResult(1),
                                                   get_value(forOp, forOp.step(), rewriter), Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), mux.getResult(1), add.getResult(2), Value());
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), mux.getResult(3), merge_condition, Value());
                }
                for (auto &op : *(forOp.getBody())) {
                    if (isa<ConstantOp, tor::YieldOp>(op)) {
                        continue;
                    }
                    Generate_operation(&op, rewriter);
                    for (auto pair : hec_operation) {
                        if (pair.first.first->getParentOp() == forOp) {
                            if (!pair.first.second) {
                                //FIXME: only consider inner loop
                                if (isa<tor::ForOp>(op)) {
                                    hec_operation[std::make_pair(forOp, pair.first.second)] = pair.second;
                                }
                                continue;
                            }
                            if (pair.first.second.val.isa<BlockArgument>()) {
                                continue;
                            }
                            /*if (pair.first.second.val.getDefiningOp()->getParentOp() != forOp) {
                                continue;
                            }*/
                            if (pair.first.second.val.getDefiningOp() != pair.first.first) {
                                continue;
                            }
                            hec_operation[std::make_pair(forOp, pair.first.second)] = pair.second;
                        }
                    }
                }
                auto yield = forOp.getBody()->getTerminator();
                for (unsigned idx = 0; idx < yield->getNumOperands(); ++idx) {
                    auto ret = yield->getOperand(idx);
                    rewriter.create<hec::AssignOp>(forOp.getLoc(), mux_set[idx + 1]->getResult(1),
                                                   get_value(forOp, ret, rewriter), Value());
                    hec_operation[std::make_pair(forOp, forOp->getResult(idx))] = branch_set[idx + 1]->getResult(3);
                }
                rewriter.create<hec::AssignOp>(forOp.getLoc(), c_merge.getResult(1),
                                               get_value(forOp, control_val.val, rewriter), Value());
                hec_operation[std::make_pair(forOp, control_val)] = branch.getResult(3);

            } else if (auto ifOp = dyn_cast<tor::IfOp>(op)) {
                //FIXME: condition cannot be used by other operation
                llvm::SmallVector<Operation *, 4> mux_set;
                llvm::SmallVector<Operation *, 4> branch_set;
                for (auto val : liveins[ifOp]) {
                    if (val && val.val == ifOp.condition()) {
                        continue;
                    }
                    std::cerr << "!!!!!";
                    val.dump();
                    llvm::SmallVector<mlir::Type, 4> types;
                    types.push_back(rewriter.getI1Type());
                    for (int loop = 0; loop < 3; ++loop) {
                        types.push_back(val ? val.val.getType() : rewriter.getIntegerType(1000));
                    }
                    auto branch = create_primitive(ifOp.getLoc(), types, "branch", "b_", rewriter);
                    branch_set.push_back(branch);
                    rewriter.create<hec::AssignOp>(ifOp.getLoc(), branch->getResult(0),
                                                   get_value(ifOp, ifOp.condition(), rewriter), Value());
                    rewriter.create<hec::AssignOp>(ifOp.getLoc(), branch->getResult(1),
                                                   get_value(ifOp, val.val, rewriter), Value());
                }
                for (auto ret : ifOp->getResults()) {
                    llvm::SmallVector<mlir::Type, 4> types;
                    for (int loop = 0; loop < 3; ++loop) {
                        types.push_back(ret.getType());
                    }
                    types.push_back(rewriter.getI1Type());
                    auto cmerge = create_primitive(ifOp.getLoc(), types, "mux_dynamic", "m_", rewriter);
                    mux_set.push_back(cmerge);
                    hec_operation[std::make_pair(ifOp, ret)] = cmerge.getResult(2);
                    rewriter.create<hec::AssignOp>(ifOp.getLoc(), cmerge.getResult(3),
                                                   get_value(ifOp, ifOp.condition(), rewriter), Value());
                }
                for (unsigned idx = 0; idx < ifOp->getNumRegions(); ++idx) {
                    int branch_count = 0;
                    for (auto val : liveins[ifOp]) {
                        if (val && val.val == ifOp.condition()) {
                            continue;
                        }
                        hec_operation[std::make_pair(ifOp, val.val)] = branch_set[branch_count++]->getResult(idx + 2);
                    }
                    if (ifOp.getRegion(idx).empty()) {
                        continue;
                    }
                    if (!ifOp.getRegion(idx).empty()) {
                        for (auto &op : *(ifOp.getBody(idx))) {
                            if (isa<ConstantOp, tor::YieldOp>(op)) {
                                continue;
                            }
                            Generate_operation(&op, rewriter);
                            for (auto pair : hec_operation) {
                                if (pair.first.first == &op && pair.first.first->getParentOp() == ifOp) {
                                    if (!pair.first.second) {
                                        continue;
                                    }
                                    if (pair.first.second.val.isa<BlockArgument>()) {
                                        continue;
                                    }
/*
                                if (pair.first.second.val.getDefiningOp()->getParentOp() != ifOp) {
                                    continue;
                                }
*/
                                    if (pair.first.second.val.getDefiningOp() != pair.first.first) {
                                        continue;
                                    }
                                    hec_operation[std::make_pair(ifOp, pair.first.second)] = pair.second;
                                }
                            }
                        }
                    }
                    auto yield = ifOp.getBody(idx)->getTerminator();
                    for (unsigned yield_cnt = 0; yield_cnt < yield->getNumOperands(); ++yield_cnt) {
                        auto ret = yield->getOperand(yield_cnt);
//                        std::cerr << idx << "::";
                        rewriter.create<hec::AssignOp>(ifOp.getLoc(), mux_set[yield_cnt]->getResult(!idx),
                                                       get_value(ifOp, ret, rewriter), Value());
                    }
                }
            } else if (auto whileOp = dyn_cast<tor::WhileOp>(op)) {
                assert(false && "While operation not finished");
            } else if (auto callOp = dyn_cast<tor::CallOp>(op)) {
                auto callee = callOp.callee();
                std::cerr << callee.str() << std::endl;
                llvm::SmallVector<mlir::Type, 4> types;
                for (auto arg : callOp.getArgOperands())
                    types.push_back(arg.getType());
                types.push_back(rewriter.getI1Type());

                for (auto ret : callOp->getResults())
                    types.push_back(ret.getType());
                types.push_back(rewriter.getI1Type());

                auto lastInsertion = rewriter.saveInsertionPoint();
                rewriter.setInsertionPoint(TopComp->getGraphOp());
                int count = 0;
                if (instanceCounts.find(callee.str()) != instanceCounts.end()) {
                    count = instanceCounts[callee.str()] + 1;
                }
                instanceCounts[callee.str()] = count;
                auto instance = rewriter.create<hec::InstanceOp>(op->getLoc(), mlir::TypeRange(types),
                                                                 callee.str() + "_" + std::to_string(count), callee);
                rewriter.restoreInsertionPoint(lastInsertion);
                for (auto ret : callOp->getResults()) {
                    llvm::SmallVector<mlir::Type, 4> types;
                    types.push_back(
                            instance.getResult(callOp.getArgOperands().size() + 1 + ret.getResultNumber()).getType());
                    types.push_back(
                            instance.getResult(callOp.getArgOperands().size() + 1 + ret.getResultNumber()).getType());
                    auto fifo = create_primitive(op->getLoc(), types, "fifo:1", "fifo_",
                                                 rewriter);
                    rewriter.create<hec::AssignOp>(op->getLoc(), fifo.getResult(0), instance.getResult(
                            callOp.getArgOperands().size() + 1 + ret.getResultNumber()), Value());
                    hec_operation[std::make_pair(op, ret)] = fifo.getResult(1);
//                    hec_operation[std::make_pair(op, ret)] = instance.getResult(
//                            callOp.getArgOperands().size() + 1 + ret.getResultNumber());
                }
//                TopComp->dump();
                instance->dump();
                instance.getReferencedComponent()->setAttr("interfc",
                                                           StringAttr::get(rewriter.getContext(), "wrapped"));
                //FIXME: void function needs control_signal
                for (unsigned idx = 0; idx < callOp.getArgOperands().size(); ++idx) {
                    rewriter.create<hec::AssignOp>(callOp.getLoc(), instance->getResult(idx),
                                                   get_value(callOp, callOp.getArgOperands()[idx], rewriter), Value());
                }
            } else if (auto loadOp = dyn_cast<tor::LoadOp>(op)) {
                assert(loadOp.indices().size() == 1 && "Invalid load operation");
                llvm::SmallVector<mlir::Type, 4> types;
                types.push_back(loadOp.indices()[0].getType());
                auto allocOp = cast<tor::AllocOp>(loadOp.memref().getDefiningOp());
                types.push_back(allocOp.getType().getElementType());
                types.push_back(loadOp.indices()[0].getType());
                types.push_back(allocOp.getType().getElementType());
                types.push_back(rewriter.getIntegerType(1000));
                auto load = create_primitive(loadOp.getLoc(), types,
                                             "load#" + std::to_string(allocOp.getType().getShape()[0]), "load_",
                                             rewriter);
                hec_operation[std::make_pair(loadOp, Liveness(loadOp.getResult()))] = load.getResult(1);

                rewriter.create<hec::AssignOp>(loadOp.getLoc(), load.getResult(0),
                                               get_value(loadOp, loadOp.indices()[0], rewriter), Value());
                rewriter.create<hec::AssignOp>(loadOp.getLoc(), load.getResult(4),
                                               get_value(loadOp, control_signal, rewriter), Value());
                auto memOp = memSet[loadOp.memref().getDefiningOp()];
                auto const &vec = loadSet[loadOp.memref().getDefiningOp()];
                for (unsigned idx = 0; idx != vec.size(); ++idx) {
                    auto const &op = vec[idx];
                    if (op == loadOp) {
                        rewriter.create<hec::AssignOp>(loadOp.getLoc(), memOp->getResult(2 * idx), load.getResult(2),
                                                       Value());
                        rewriter.create<hec::AssignOp>(loadOp.getLoc(), load.getResult(3),
                                                       memOp->getResult(2 * idx + 1), Value());
                    }
                }
            } else if (auto storeOp = dyn_cast<tor::StoreOp>(op)) {
                assert(storeOp.indices().size() == 1 && "Invalid load operation");
                llvm::SmallVector<mlir::Type, 4> types;
                types.push_back(storeOp.indices()[0].getType());
                auto allocOp = cast<tor::AllocOp>(storeOp.memref().getDefiningOp());
                types.push_back(allocOp.getType().getElementType());
                types.push_back(storeOp.indices()[0].getType());
                types.push_back(allocOp.getType().getElementType());
                types.push_back(rewriter.getIntegerType(1000));
                auto store = create_primitive(storeOp.getLoc(), types,
                                              "store#" + std::to_string(allocOp.getType().getShape()[0]), "store_",
                                              rewriter);
                rewriter.create<hec::AssignOp>(storeOp.getLoc(), store.getResult(0),
                                               get_value(storeOp, storeOp.indices()[0], rewriter), Value());
                rewriter.create<hec::AssignOp>(storeOp.getLoc(), store.getResult(1),
                                               get_value(storeOp, storeOp.value(), rewriter), Value());
                rewriter.create<hec::AssignOp>(storeOp.getLoc(), store.getResult(4),
                                               get_value(storeOp, control_signal, rewriter), Value());
                auto memOp = memSet[storeOp.memref().getDefiningOp()];
                unsigned loadSize = loadSet[storeOp.memref().getDefiningOp()].size();
                auto const &vec = storeSet[storeOp.memref().getDefiningOp()];
                for (unsigned idx = 0; idx != vec.size(); ++idx) {
                    auto const &op = vec[idx];
                    if (op == storeOp) {
                        rewriter.create<hec::AssignOp>(storeOp.getLoc(), memOp->getResult(2 * (idx + loadSize)),
                                                       store.getResult(2), Value());
                        rewriter.create<hec::AssignOp>(storeOp.getLoc(), memOp->getResult(2 * (idx + loadSize) + 1),
                                                       store.getResult(3), Value());
                    }
                }
            } else {
#define CREATE_PRIMITIVE(TYPE, PRIMITIVE, NAME) \
                if (auto tor_op = dyn_cast<TYPE>(op)) { \
                    llvm::SmallVector<mlir::Type, 4> types; \
                    for (auto val : tor_op->getOperands()) { \
                        types.push_back(val.getType());  \
                    }                \
                    for (auto val : tor_op->getResults()) {\
                        types.push_back(val.getType());  \
                    }                \
                    auto primitive = create_primitive(op->getLoc(), types, PRIMITIVE, NAME, rewriter); \
                    int retNum = tor_op->getNumOperands();  \
                    for (auto ret : tor_op->getResults()) { \
                        hec_operation[std::make_pair(tor_op, Liveness(ret))] = primitive.getResult(retNum++); \
                    }\
                    int primitive_count = 0;             \
                    for (auto val : tor_op->getOperands()) {\
                        rewriter.create<hec::AssignOp>(tor_op.getLoc(), primitive.getResult(primitive_count++), get_value(op, val, rewriter), Value()); \
                    }                           \
                }
                CREATE_PRIMITIVE(tor::AddIOp, "add_integer", "addi_")
                else CREATE_PRIMITIVE(tor::SubIOp, "sub_integer", "subi_")
                else CREATE_PRIMITIVE(tor::MulIOp, "mul_integer", "muli_")
                else CREATE_PRIMITIVE(tor::AddFOp, "add_float", "addf_")
                else CREATE_PRIMITIVE(tor::SubFOp, "sub_float", "subf_")
                else CREATE_PRIMITIVE(tor::MulFOp, "mul_float", "mulf_")
                else CREATE_PRIMITIVE(tor::DivFOp, "div_float", "divf_")
                else CREATE_PRIMITIVE(TruncateIOp, "trunc_integer", "trunci_")
                else CREATE_PRIMITIVE(tor::CmpIOp,
                                      std::string("cmp_integer_") + tor::stringifyEnum(tor_op.predicate()).str(),
                                      "cmpi_")
                else CREATE_PRIMITIVE(tor::CmpFOp,
                                      std::string("cmp_float_") + tor::stringifyEnum(tor_op.predicate()).str(),
                                      "cmpf_")
                else CREATE_PRIMITIVE(ShiftLeftOp, "shift_left", "shr_")
                else CREATE_PRIMITIVE(SelectOp, "select", "select_")
                else CREATE_PRIMITIVE(NegFOp, "neg_float", "negf_")
                else CREATE_PRIMITIVE(AndOp, "and", "and_")
                else CREATE_PRIMITIVE(FPToSIOp, "fptosi", "fptosi_")
                else {
                    op->dump();
                    assert(false && "Invalid operation in dynamic schedule");
                }

#undef CREATE_PRIMITIVE
                for (auto val : op->getOperands()) {
//                    get_value(op, val).
                    auto tmp = get_value(op, val, rewriter);
                    if (tmp.isa<OpResult>()) {
                        auto result = tmp.cast<OpResult>();
                        std::cerr << result.getResultNumber() << " :: ";
                    }
                    tmp.dump();
                }
            }
            std::cerr << "Finished: ";
            op->dump();
        }

        void insert_fifo(hec::ComponentOp &component, PatternRewriter &rewriter) {
            std::map<Operation *, std::vector<std::pair<Operation *, Operation *>>> graph;
            auto getLatency = [&](Operation *op) {
                if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
                    std::string primName = primitive.primitiveName().str();
                    if (primName == "mul_float") {
                        return 9;
                    }
                    if (primName == "add_float") {
                        return 13;
                    }
                    if (primName == "sub_float") {
                        return 13;
                    }
                    if (primName == "mul_integer") {
                        return 2;
                    }
                    if (primName == "div_float") {
                        return 30;
                    }
                } else if (auto instanceOp = dyn_cast<hec::InstanceOp>(op)) {
                    int latency = instanceOp.getReferencedComponent()->getAttr("latency").cast<IntegerAttr>().getInt();
                    return latency;
                }
                return 0;
            };

            std::map<Operation *, int> time;
            for (auto &op : *(component.getBody())) {
                if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
                    std::string primName = primitive.primitiveName().str();
                    if (primName == "mux_dynamic" || primName == "control_merge") {
                        time[primitive] = 0;
                    } else if (primName.find("dyn_Mem") != std::string::npos) {
                        time[primitive] = 0;
                    } else {
                        time[primitive] = -1;
                    }
                } else if (auto instance = dyn_cast<hec::InstanceOp>(op)) {
                    time[instance] = -1;
                }
            }
            for (auto &op : *(component.getGraphOp().getBody())) {
                if (auto assignOp = dyn_cast<hec::AssignOp>(op)) {
                    if (assignOp.src().isa<BlockArgument>()) {
                        time[assignOp.dest().getDefiningOp()] = 0;
                        continue;
                    }
                    auto src = assignOp.src();
                    auto dest = assignOp.dest();
                    graph[dest.getDefiningOp()].push_back(std::make_pair(src.getDefiningOp(), &op));
                }
            }
            std::function<int(Operation *)> get_schedule = [&](Operation *op) {
                if (time[op] != -1) {
                    return time[op];
                }
                int now_time = 0;
                for (auto pred : graph[op]) {
                    now_time = std::max(now_time, get_schedule(pred.first));
                }
                return time[op] = now_time + getLatency(op);
            };
            for (const auto &pair : time) {
                get_schedule(pair.first);
            }
            for (const auto &pair : time) {
                int end = time[pair.first];
                if (end == 0) {
                    continue;
                }
                pair.first->dump();
                int start = end - getLatency(pair.first);
                for (auto pred : graph[pair.first]) {
                    int temp = end - time[pred.first];
                    if (temp != 0) {
                        auto assignOp = cast<hec::AssignOp>(pred.second);
                        std::cerr << temp;
                        assignOp.dump();
                        auto src = assignOp.src();
                        auto dest = assignOp.dest();
                        llvm::SmallVector<mlir::Type, 4> types;
                        types.push_back(src.getType());
                        types.push_back(dest.getType());
                        auto fifo = create_primitive(component.getLoc(), types, "fifo:" + std::to_string(temp), "fifo_",
                                                     rewriter);
                        rewriter.create<hec::AssignOp>(component.getLoc(), fifo.getResult(0), src, Value());
                        rewriter.create<hec::AssignOp>(component.getLoc(), dest, fifo.getResult(1), Value());
//                        component->dump();
                        rewriter.eraseOp(assignOp);
                    }
                }
            }
//            exit(-1);
//            for (auto &op : *(component.getGraphOp().getBody())) {
//                if (auto primitive = dyn_cast<hec::AssignOp>(op)) {
//                    if (primitive.src().isa<BlockArgument>()) {
//                        continue;
//                    }
//                    graph[primitive.dest().getDefiningOp()].push_back(primitive.src().getDefiningOp());
//                    rev_graph[primitive.src().getDefiningOp()].push_back(primitive.dest().getDefiningOp());
//                }
//            }

        }

        void Generate_operation(tor::FuncOp funcOp, PatternRewriter &rewriter) {
            auto context = funcOp.getContext();
            mlir::StringAttr name = funcOp.getNameAttr();
            llvm::SmallVector<mlir::hec::ComponentPortInfo, 16> ports;
            mlir::StringAttr interfc = mlir::StringAttr::get(context, "wrapped");
            mlir::StringAttr style = mlir::StringAttr::get(context, "handshake");
            auto funcType = funcOp.getType();
//            auto funcArgs = funcOp.getArguments();
            size_t icount = 0;
            for (auto inPort : funcType.getInputs()) {
                llvm::StringRef port_name(std::string("in") + std::to_string(icount++));

                ports.push_back(
                        hec::ComponentPortInfo(
                                mlir::StringAttr::get(context, port_name),
                                inPort, hec::PortDirection::INPUT));
                std::string typeStr;
                llvm::raw_string_ostream stro(typeStr);
                ports.back().type.print(stro);
                stro.flush();
                std::cerr << typeStr << std::endl;
            }
            ports.push_back(hec::ComponentPortInfo(mlir::StringAttr::get(context, "in"), rewriter.getIntegerType(1000),
                                                   hec::PortDirection::INPUT));
            size_t ocount = 0;
            for (auto outPort : funcType.getResults()) {
                llvm::StringRef port_name(std::string("out") + std::to_string(ocount++));

                ports.push_back(
                        hec::ComponentPortInfo(
                                mlir::StringAttr::get(context, port_name),
                                outPort, hec::PortDirection::OUTPUT));
                std::string typeStr;
                llvm::raw_string_ostream stro(typeStr);
                ports.back().type.print(stro);
                stro.flush();
                std::cerr << typeStr << std::endl;
            }
            ports.push_back(hec::ComponentPortInfo(mlir::StringAttr::get(context, "out"), rewriter.getIntegerType(1000),
                                                   hec::PortDirection::OUTPUT));
            auto component = rewriter.create<hec::ComponentOp>(funcOp.getLoc(), name, ports, interfc, style);
            TopComp = &component;
            if (component.getGraphOp().body().empty())
                component.getGraphOp().body().push_back(new mlir::Block);
            for (unsigned idx = 0; idx != funcOp.getNumArguments(); ++idx) {
                hec_operation[std::make_pair(funcOp, funcOp.getArgument(idx))] = component.getArgument(idx);
            }
            hec_operation[std::make_pair(funcOp, control_signal)] = component.getArgument(funcOp.getNumArguments());
            rewriter.setInsertionPointToStart(component.getBody());

            std::cerr << "-----Memory op--------\n";
            //FIXME : How to deal with complex dependence for memory operation
            for (auto const &pair : loadstoreSet) {
                int loadsize = loadSet[pair.first].size();
                int storesize = storeSet[pair.first].size();
//                    std::cerr << pair.second.size() << ":" << loadsize << "," << storesize;
                pair.first->dump();
                llvm::SmallVector<mlir::Type, 4> types;
                std::string memInfo = "";
                for (auto const &loadOp : loadSet[pair.first]) {
                    types.push_back(cast<tor::LoadOp>(loadOp).indices()[0].getType());
                    types.push_back(cast<tor::AllocOp>(pair.first).getType().getElementType());
                    memInfo = "#" + std::to_string(cast<tor::AllocOp>(pair.first).getType().getShape()[0]);
                }
                for (auto const &storeOp : storeSet[pair.first]) {
                    types.push_back(cast<tor::StoreOp>(storeOp).indices()[0].getType());
                    types.push_back(cast<tor::AllocOp>(pair.first).getType().getElementType());
                    memInfo = "#" + std::to_string(cast<tor::AllocOp>(pair.first).getType().getShape()[0]);
                }
                memSet[pair.first] = create_primitive(funcOp.getLoc(), types,
                                                      "dyn_Mem:" + std::to_string(loadsize) + "," +
                                                      std::to_string(storesize) + memInfo, "mem_", rewriter);
//                memOp->dump();
            }
//            component->dump();
//            exit(-1);
//            for (auto val : liveins[funcOp]) {
//
//            }
            rewriter.setInsertionPointToEnd(&(component.getGraphOp().body().front()));
            for (auto &op : funcOp.getRegion().front()) {
                if (isa<tor::TimeGraphOp, tor::ReturnOp>(op)) {
                    continue;
                }
                if (auto constant = dyn_cast<ConstantOp>(op)) {
                    assert(false && "Invalid constant operation");
                }
                Generate_operation(&op, rewriter);
                for (auto pair : hec_operation) {
                    if (pair.first.first->getParentOp() == funcOp) {
                        if (!pair.first.second) {
                            continue;
                        }
                        if (pair.first.second.val.isa<BlockArgument>()) {
                            continue;
                        }
                        if (pair.first.second.val.getDefiningOp()->getParentOp() != funcOp) {
                            continue;
                        }
                        hec_operation[std::make_pair(funcOp, pair.first.second)] = pair.second;
                    }
                }
            }
            auto returnOp = funcOp.getBodyBlock()->getTerminator();
            for (unsigned idx = 0; idx < returnOp->getNumOperands(); ++idx) {
                auto ret = returnOp->getOperand(idx);
//                ret.dump();
                rewriter.create<hec::AssignOp>(funcOp.getLoc(),
                                               component.getArgument(component.numInPorts() + idx),
                                               get_value(funcOp, ret, rewriter), Value());
            }
//            component->dump();
//            exit(-1);
            for (auto &op : *(component.getBody())) {
                if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
                    auto portInfo = primitive.getPrimitivePortInfo();
                    for (unsigned idx = 0; idx < portInfo.size(); ++idx) {
                        if (portInfo[idx].direction == hec::PortDirection::OUTPUT) {
                            auto port = primitive.getResult(idx);
                            std::vector<hec::AssignOp> assign_set;
                            int use_count = 0;
                            for (auto &bval : port.getUses()) {
                                if (auto assign = dyn_cast<hec::AssignOp>(bval.getOwner())) {
                                    if (assign.src() == port) {
                                        assign_set.push_back(assign);
                                        ++use_count;
                                    }
                                }
                            }
                            if (use_count > 1) {
//                                std::cerr << primitive.primitiveName().str() << "?";
//                                primitive.dump();
//                                std::cerr << portInfo[idx].name.getValue().str() << ":";
//                                port.dump();
                                llvm::SmallVector<mlir::Type, 4> types;
                                for (int loop = 0; loop < use_count + 1; ++loop) {
                                    types.push_back(port.getType());
                                }
                                auto fork = create_primitive(funcOp.getLoc(), types,
                                                             "fork:" + std::to_string(use_count), "f_", rewriter);
                                use_count = 0;
                                for (auto assign : assign_set) {
                                    assign.setOperand(1, fork.getResult(++use_count));
                                }
                                rewriter.create<hec::AssignOp>(funcOp.getLoc(), fork->getResult(0),
                                                               port, Value());
                            } else if (use_count == 0) {
//                                std::cerr << primitive.primitiveName().str() << "SINK";
//                                primitive.dump();
//                                std::cerr << portInfo[idx].name.getValue().str() << ":";
//                                port.dump();
                            }
                        }
                    }
                }
            }
//            component->dump();
//            exit(-1);

            insert_fifo(component, rewriter);

//            for (auto &op : *(component.getBody())) {
//                if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
//                    auto portInfo = primitive.getPrimitivePortInfo();
//                    for (unsigned idx = 0; idx < portInfo.size(); ++idx) {
//                        if (portInfo[idx].direction == hec::PortDirection::OUTPUT) {
//                            auto port = primitive.getResult(idx);
//                            if (!port.hasOneUse() && !port.use_empty()) {
//                                std::cerr << portInfo[idx].name.getValue().str() << "!!!";
//                                port.dump();
//                                assert(false && "Need Insert Fork");
//                            }
//                        }
//                    }
//                }
//            }
        }

        void generate_top(tor::FuncOp funcOp, PatternRewriter &rewriter) {
            rewriter.setInsertionPointAfter(funcOp->getParentOp());
            bool found = false;
            auto moduleOp = cast<ModuleOp>(funcOp->getParentOp()->getParentOp());
            hec::DesignOp hecDesign;
            for (auto &op : *(moduleOp.getBody())) {
                if (auto hecdesignOp = dyn_cast<hec::DesignOp>(op)) {
                    found = true;
                    hecDesign = hecdesignOp;
                    break;
                }
            }
            if (!found) {
                hecDesign = rewriter.create<hec::DesignOp>(funcOp.getLoc(),
                                                           cast<tor::DesignOp>(funcOp->getParentOp()).symbol());
            }
            if (hecDesign.body().empty())
                hecDesign.body().push_back(new mlir::Block);
            rewriter.setInsertionPointToStart(hecDesign.getBody());

            for (auto &op : (funcOp->getParentOp())->getRegion(0).front()) {
                if (auto constant = dyn_cast<ConstantOp>(op)) {
                    new_operation[constant] = rewriter.clone(*constant);
                } else if (auto allocOp = dyn_cast<tor::AllocOp>(op)) {

                }
            }
            control_signal = Value();
            rewriter.setInsertionPointToEnd(hecDesign.getBody());
            Generate_operation(funcOp, rewriter);

//            funcOp->getParentOp()->getParentOp()->dump();
        }

        struct DynamicSchedule : public OpRewritePattern<tor::DesignOp> {
            DynamicSchedule(MLIRContext *context) : OpRewritePattern<tor::DesignOp>(context, 1) {}

            LogicalResult
            matchAndRewrite(tor::DesignOp designOp, PatternRewriter &rewriter) const override {
                instanceCounts.clear();
                tor::FuncOp funcOp;
                for (auto &op : *(designOp.getBody())) {
                    if (auto sop = dyn_cast<tor::FuncOp>(op)) {
                        if (sop.getName() == "main") {
                            funcOp = sop;
                        } else {
                            sop->dump();
                        }
                    }
                }
                if (funcOp->hasAttr("strategy")) {
                    if (auto str = funcOp->getAttr("strategy").dyn_cast<StringAttr>()) {
                        if (str.getValue() == "dynamic") {
                            if (funcOp->hasAttr("Dynamic")) {
                                return failure();
                            } else {
                                funcOp->setAttr("Dynamic", StringAttr::get(getContext(), "Done"));
                            }
                        } else {
                            return failure();
                        }
                    }
                } else {
//                    assert(false && "Undefined strategy");
                }

                std::cerr << "------Analysis--------\n";
                analysis_top(funcOp);

                std::cerr << "---Memory  analysis---\n";
                funcOp.walk([&](Operation *op) {
                    if (auto loadOp = dyn_cast<tor::LoadOp>(op)) {
                        auto memref = loadOp.memref();
                        loadSet[memref.getDefiningOp()].push_back(loadOp);
                        loadstoreSet[memref.getDefiningOp()].push_back(loadOp);
                    } else if (auto storeOp = dyn_cast<tor::StoreOp>(op)) {
                        auto memref = storeOp.memref();
                        storeSet[memref.getDefiningOp()].push_back(storeOp);
                        loadstoreSet[memref.getDefiningOp()].push_back(storeOp);
                    }
                });

                /*std::cerr << "------Load ops--------\n";
                for (auto pair : loadSet) {
                    pair.first->dump();
                    for (auto op : pair.second) {
                        op->dump();
                    }
                }
                std::cerr << "-----Store ops--------\n";
                for (auto pair : storeSet) {
                    pair.first->dump();
                    for (auto op : pair.second) {
                        op->dump();
                    }
                }*/

                std::cerr << "------Generate--------\n";
                generate_top(funcOp, rewriter);
                Operation *torDesign = funcOp->getParentOp();
                rewriter.eraseOp(designOp);
//                rewriter.eraseOp(funcOp);

                return success();
            }
        };

#undef TIME_NODE
    }

    struct DynamicSchedulePass : public dynamicScheduleBase<DynamicSchedulePass> {
        void runOnOperation() override {
            mlir::ModuleOp m = getOperation();

            if (m.walk([&](tor::DesignOp op) {
                        mlir::RewritePatternSet patterns(&getContext());
                        patterns.insert<dynamic::DynamicSchedule>(op.getContext());

                        if (failed(applyOpPatternsAndFold(op, std::move(patterns))))
                            return WalkResult::advance();
//                            return WalkResult::interrupt();

                        return WalkResult::advance();
                    })
                    .wasInterrupted()) {
                return signalPassFailure();
            }
//            for (auto &module : *(m.getBody())) {
//                if(auto design = dyn_cast<tor::DesignOp>(module)) {
//                    for (auto &torFunc : design.body().front()) {
//                        if(torFunc.getAttr("strategy").cast<StringAttr>().getValue() == "dynamic") {
//                            torFunc.dump();
//                            for (auto &operation : torFunc.getRegion(0).front()) {
//                                operation.dump();
//                            }
//                            Liveness live(&torFunc);
//                            live.dump();
//                        }
//                    }
//                }
//            }
        }
    };

    std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
    createDynamicSchedulePass() {
        return std::make_unique<DynamicSchedulePass>();
    }

} // namespace mlir
