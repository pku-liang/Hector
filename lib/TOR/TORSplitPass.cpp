#include "TOR/PassDetail.h"
#include "mlir/Analysis/Utils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "TOR/TOR.h"
#include "TOR/TORDialect.h"

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

#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>

#define DEBUG_TYPE "split-schedule"

namespace mlir {
    namespace split {
#define TIME_NODE 1005

        struct TimeEdge {
            int from;
            int to;
            bool loop_exit;
            bool Static;
            bool valid;
            bool pipeline;
            Attribute attr;
            std::vector<Operation *> ops;

            TimeEdge(int _from, int _to, Attribute _attr, bool _static = true, bool _exit = false, bool _pipe = false) {
                from = _from;
                to = _to;
                attr = _attr;
                loop_exit = _exit;
                ops.clear();
                Static = _static;
                valid = true;
                pipeline = _pipe;
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

        bool connected(int x, int end) {
            if (x == end)
                return true;
            for (auto &edge : timeGraph[x]) {
                if (edge.Static && edge.valid) {
                    if (connected(edge.to, end))
                        return true;
                }
            }
            return false;
        };

        bool strongConnected(int x, int end) {
            if (x == end)
                return true;
            for (auto &edge : timeGraph[x]) {
                if (edge.Static && edge.valid) {
                    if (!strongConnected(edge.to, end))
                        return false;
                } else return false;
            }
            return true;
        };

        void bind_operation(int src, int dest, Operation *op) {
            for (auto &edge : timeGraph[src]) {
                if (edge.to == dest || connected(edge.to, dest)) {
                    edge.ops.push_back(op);
                }
            }
        }

        struct SplitSchedule : public OpRewritePattern<tor::FuncOp> {
            SplitSchedule(MLIRContext *context) : OpRewritePattern<tor::FuncOp>(context, 1) {}

            LogicalResult
            matchAndRewrite(tor::FuncOp funcOp, PatternRewriter &rewriter) const override {
                if (funcOp->hasAttr("strategy")) {
                    if (auto str = funcOp->getAttr("strategy").dyn_cast<StringAttr>()) {
                        if (str.getValue() == "mixed") {
                        } else {
                            return failure();
                        }
                    }
                } else {
                    funcOp->setAttr("strategy", StringAttr::get(getContext(), "mixed"));
                }
                for (int i = 0; i < TIME_NODE; i++) {
                    timeGraph[i].clear();
                    //                succOp[i].clear();
                }
                uint32_t MAX_INDEX = 0;
                uint32_t START, END;
                tor::TimeGraphOp timeGraphOp;
                for (auto &block : funcOp) {
                    for (auto &op : block) {
                        if (auto timegraph = dyn_cast<tor::TimeGraphOp>(op)) {
                            timeGraphOp = timegraph;
                        }
                    }
                }
                timeGraphOp->dump();
                bool foundStatic = false;
                bool foundDynamic = false;
                bool foundPipeline = false;
                START = timeGraphOp.starttime();
                END = timeGraphOp.endtime();
                MAX_INDEX = std::max(START, END);
                for (auto &block : timeGraphOp.region()) {
                    for (auto &op : block) {
                        op.dump();
                        if (auto succ = dyn_cast<tor::SuccTimeOp>(op)) {
                            MAX_INDEX = std::max(MAX_INDEX, succ.time());
                            for (unsigned i = 0; i < succ.points().size(); i++) {
                                auto from = succ.points()[i];
                                auto comp_edge = succ.edges()[i].cast<DictionaryAttr>();
                                bool pipeline = comp_edge.get("pipeline").operator bool();
                                if (pipeline) {
                                    succ->dump();
                                }
                                foundPipeline |= pipeline;
                                auto edge_info = comp_edge.get("type");
                                int index = from.cast<IntegerAttr>().getInt();
                                auto info = edge_info.cast<StringAttr>().getValue().str();
                                if (info.find("dynamic") != StringRef::npos) {
                                    foundDynamic = true;
                                    timeGraph[index].push_back(
                                            TimeEdge(index, succ.time(), succ.edges()[i], false,
                                                     info.find("for") != StringRef::npos ||
                                                     info.find("while") != StringRef::npos, pipeline));
                                } else if (info.find("static") != StringRef::npos) {
                                    foundStatic = true;
                                    timeGraph[index].push_back(
                                            TimeEdge(index, succ.time(), succ.edges()[i], true,
                                                     info.find("for") != StringRef::npos ||
                                                     info.find("while") != StringRef::npos, pipeline));
                                } else {
                                    edge_info.dump();
                                    assert("Unexpected edge_info attribute" && false);
                                }
                                std::cerr << "???" << succ.time() << std::endl;
                                succOp[succ.time()] = &op;
                            }
                        }
                    }
                }
                if (!foundStatic && !foundPipeline) {
                    funcOp->setAttr("strategy", StringAttr::get(getContext(), "dynamic"));
                    return failure();
                }
                if (!foundDynamic && !foundPipeline) {
                    funcOp->setAttr("strategy", StringAttr::get(getContext(), "static"));
                    return failure();
                }
                if (foundDynamic) {
                    funcOp->setAttr("strategy", StringAttr::get(getContext(), "dynamic"));
                } else {
                    funcOp->setAttr("strategy", StringAttr::get(getContext(), "static"));
                }
                std::cerr << "~~~~~~~~~~~~~~~~~~~~~~~\n";
                memset(ifEnd, -1, sizeof(ifEnd));
                memset(ifBegin, -1, sizeof(ifBegin));
                memset(whileEnd, -1, sizeof(whileEnd));
                memset(forEnd, -1, sizeof(forEnd));
                memset(whileBegin, -1, sizeof(whileBegin));
                memset(forBegin, -1, sizeof(forBegin));
                funcOp.walk([&](Operation *op) {
#define BIND(OpType)                   \
if (auto sop = dyn_cast<OpType>(op)) \
bind_operation(sop.starttime(), sop.endtime(), op);
                    BIND(tor::AddIOp)
                    BIND(tor::SubIOp)
                    BIND(tor::MulIOp)
                    BIND(tor::CmpIOp)
                    BIND(tor::AddFOp)
                    BIND(tor::SubFOp)
                    BIND(tor::MulFOp)
                    BIND(tor::CmpFOp)
                    BIND(tor::LoadOp)
                    BIND(tor::StoreOp)
//                    BIND(tor::IfOp)
//                    BIND(tor::WhileOp)
//                    BIND(tor::ForOp)
#undef BIND
                    if (auto ifop = dyn_cast<tor::IfOp>(op)) {
                        ifEnd[ifop.starttime()] = ifop.endtime();
                        ifBegin[ifop.endtime()] = ifop.starttime();
                        ifOp[ifop.starttime()] = op;
                    } else if (auto whileop = dyn_cast<tor::WhileOp>(op)) {
                        whileEnd[whileop.starttime()] = whileop.endtime();
                        whileBegin[whileop.endtime()] = whileop.starttime();
                        whileOp[whileop.starttime()] = op;
                    } else if (auto forop = dyn_cast<tor::ForOp>(op)) {
                        forEnd[forop.starttime()] = forop.endtime();
                        forBegin[forop.endtime()] = forop.starttime();
                        forOp[forop.starttime()] = op;
//                        std::cerr << ":," << forop.starttime() << forop.endtime() << std::endl;
                    }
                });
//                exit(0);
                for (uint32_t i = 0; i < MAX_INDEX; i++) {
                    std::cerr << i << ": ";
                    for (auto &edge : timeGraph[i]) {
                        std::cerr << "<" << edge.to << "," << edge.Static; //<< std::endl;
                        //                    for (auto sop : edge.ops) {
                        //                        std::cerr << "\t";
                        //                        sop->dump();
                        //                    }
                        std::cerr << ">; ";
                        //                    std::cerr << std::endl;
                    }
                    std::cerr << std::endl;
                }
//                for (uint32_t i = 0; i < MAX_INDEX; i++) {
//                    for (auto &edge : timeGraph[i]) {
//                        if (edge.loop_exit) {
//                        }
//                    }
//                }
                int visitEnd[TIME_NODE], visitEndCount = 0;
                memset(visitEnd, 0, sizeof(visitEnd));
                std::vector<TimeEdge *> eraseEdges;

                int visitMark[TIME_NODE * TIME_NODE], visitMarkCount = 0, nodeMark[TIME_NODE];
                memset(visitMark, 0, sizeof(visitMark));
                memset(nodeMark, 0, sizeof(nodeMark));
                std::function<void(int, int)> markStatic = [&](int start, int end) {
                    std::cerr << "!!!" << start << " , " << end << std::endl;
                    if (visitMark[start * TIME_NODE + end] == visitMarkCount) {
                        return;
                    }
                    visitMark[start * TIME_NODE + end] = visitMarkCount;
                    nodeMark[start] = visitMarkCount;
                    if (forEnd[start] != -1 && !connected(end, forEnd[start])) {
                        markStatic(start, forEnd[start]);
                        TimeEdge *edge = &(timeGraph[start][0]);
                        if (!edge->loop_exit) {
                            edge = &(timeGraph[start][1]);
                        }
                        if (edge->Static && edge->valid) {
                            eraseEdges.push_back(edge);
                            markStatic(edge->to, end);
                        }
                        return;
                    }
                    if (start != end) {
                        for (auto &edge : timeGraph[start]) {
                            if (edge.Static && edge.valid && connected(edge.to, end)) {
                                eraseEdges.push_back(&edge);
                                markStatic(edge.to, end);
                            }
                        }
                    }
                };
                std::set<Operation *> outline_ops;
                Operation *for_op;
                auto outline = [&](int start, int end, bool pipe = false) {
                    outline_ops.clear();
                    std::function<void(int, std::function<void(Operation *op)>)> visitOperation =
                            [&](int start, std::function<void(Operation *op)> insert) {
                                if (start == end)
                                    return;
                                std::cerr << "~~" << start << " " << end << std::endl;
                                if (ifEnd[start] != -1 && strongConnected(start, ifEnd[start])) {
                                    insert(ifOp[start]);
                                    visitOperation(ifEnd[start], insert);
                                    return;
                                }
                                if (forEnd[start] != -1 && (end == forEnd[start] || !connected(end, forEnd[start]))) {
                                    insert(forOp[start]);
                                    auto edge = timeGraph[start][0];
                                    if (!edge.loop_exit) {
                                        edge = timeGraph[start][1];
                                    }
                                    visitOperation(edge.to, insert);
                                    return;
                                }
                                for (auto &edge : timeGraph[start]) {
                                    if (edge.Static && edge.valid && connected(edge.to, end)) {
                                        for (auto &op : edge.ops) {
                                            insert(op);
                                        }
                                        visitOperation(edge.to, insert);
                                    }
                                }
                            };
                    //FIXME : Consider memory operations

                    std::cerr << "OUTLINE" << start << "," << end << std::endl;
                    visitOperation(start, [&](Operation *op) {
                        op->walk([&](Operation *op) {
                            outline_ops.insert(op);
                        });
                    });
                    struct Liveness {
                        Value val;

                        Liveness(const Value &_val) {
                            val = _val;
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
                    };
                    std::set<Liveness> liveins, liveouts;
                    for (auto op : outline_ops) {
                        op->dump();
                        for (auto val : op->getResults()) {
                            for (auto it = val.getUses().begin(); it != val.getUses().end(); ++it) {
                                auto bop = it.getUser();
                                if (outline_ops.find(bop) == outline_ops.end()) {
                                    liveouts.insert(val);
                                    std::cerr << "FOUND OUT : ";
                                    bop->dump();
                                }
                            }
                        }
                        for (unsigned idx = 0; idx != op->getNumOperands(); ++idx) {
                            auto val = op->getOperand(idx);
//                            val.dump();
                            if (auto arg = val.dyn_cast<BlockArgument>()) {
                                arg.dump();
                                std::cerr << "~~~";
                                if (outline_ops.find(arg.getOwner()->getParentOp()) != outline_ops.end()) {
                                    continue;
                                }
                                std::cerr << "FOUND Argument IN ";
                                val.dump();
                                liveins.insert(val);

                            } else {
                                auto bop = val.getDefiningOp();
                                if (isa<ConstantOp, tor::AllocOp>(bop)) {
                                    continue;
                                }
                                if (outline_ops.find(bop) == outline_ops.end()) {
                                    std::cerr << "FOUND IN ";
                                    bop->dump();
                                    std::cerr << bop->getNumResults() << "???";
                                    for (auto bval : bop->getResults()) {
                                        bval.dump();
                                        if (bval == val) {
                                            liveins.insert(val);
                                            std::cerr << bval.getResultNumber() << ":";
                                        }
                                    }
                                    bop->dump();
                                }
                            }
                        }
                    }
                    llvm::SmallVector<mlir::Type, 8> argTypes;
                    llvm::SmallVector<mlir::Type, 8> retTypes;
                    llvm::SmallVector<mlir::Value, 8> argValues;
                    llvm::SmallVector<mlir::Value, 8> retValues;
                    std::map<Liveness, int> argNum;
                    std::map<Liveness, int> retNum;

                    std::cerr << "---------IN---------" << std::endl;
                    unsigned arg_count = 0, ret_count = 0;
                    if (pipe) {
                        auto forOp = cast<tor::ForOp>(for_op);
                        auto insertArg = [&](Liveness val) {
                            val.val.dump();
                            argTypes.push_back(val.val.getType());
                            argValues.push_back(val.val);
                            argNum[val] = arg_count++;
                        };
                        liveins.insert(forOp.lowerBound());
                        liveins.insert(forOp.upperBound());
                        liveins.insert(forOp.step());
                        insertArg(Liveness(forOp.lowerBound()));
                        insertArg(Liveness(forOp.upperBound()));
                        insertArg(Liveness(forOp.step()));
                        for (auto val : liveins) {
                            if (val == forOp.lowerBound())continue;
                            if (val == forOp.upperBound())continue;
                            if (val == forOp.step())continue;
                            std::cerr << "Result :";// << val.second << ":";
                            insertArg(val);
                        }
                    } else {
                        for (auto val : liveins) {
                            std::cerr << "Result :";// << val.second << ":";
                            val.val.dump();
                            argTypes.push_back(val.val.getType());
                            argValues.push_back(val.val);
                            argNum[val] = arg_count++;
                        }
                    }

                    std::cerr << "---------OUT---------" << std::endl;
                    for (auto val : liveouts) {
                        std::cerr << "Result :";// << val.second << ":";
                        val.val.dump();
                        retTypes.push_back(val.val.getType());
                        retNum[val] = ret_count++;
                    }
                    retValues.resize(ret_count);

                    std::cerr << "---------DONE--------" << std::endl;

                    rewriter.setInsertionPoint(funcOp);
                    auto func_type = rewriter.getFunctionType(argTypes, retTypes);
                    static int func_num = 0;
                    std::string func_name = "outline_" + std::to_string(func_num++);
                    auto out_func = rewriter.create<tor::FuncOp>(funcOp.getLoc(), func_name, func_type);
                    out_func->setAttr("strategy", StringAttr::get(getContext(), "static"));
                    rewriter.createBlock(&(out_func.getBody()), {}, argTypes);

                    std::cerr << "------NEW FUNC-------" << std::endl;

                    Operation *lastOp = NULL;
                    /*for (unsigned idx = 0; idx < argValues.size(); ++idx) {
                        if (!lastOp) {
                            if (!argValues[idx].isa<BlockArgument>()) {
                                lastOp = argValues[idx].getDefiningOp();
                            } else {
                                auto blockArg = argValues[idx].dyn_cast<BlockArgument>();
                                lastOp = &(blockArg.getOwner()->front());
                            }
                            continue;
                        }
                        if (argValues[idx].isa<BlockArgument>()) {
                            continue;
                        }
                        Operation *newOp = argValues[idx].getDefiningOp();
                        if (lastOp->getBlock() == newOp->getBlock()) {
                            if (lastOp->isBeforeInBlock(newOp)) {
                                lastOp = newOp;
                            }
                        } else {
                            if (lastOp->isBeforeInBlock(lastOp->getBlock()->findAncestorOpInBlock(*newOp))) {
                                lastOp = newOp;
                            }
                        }
                    }*/
                    visitOperation(start, [&](Operation *op) {
                        if (!lastOp) {
                            lastOp = op;
                            return;
                        }
                        if (lastOp->getBlock() == op->getBlock()) {
                            if (!lastOp->isBeforeInBlock(op)) {
                                lastOp = op;
                            }
                        } else {
                            if (!lastOp->isBeforeInBlock(lastOp->getBlock()->findAncestorOpInBlock(*op))) {
                                lastOp = op;
                            }
                        }
                    });
                    rewriter.setInsertionPointAfter(lastOp);
                    auto callOp = rewriter.create<tor::CallOp>(lastOp->getLoc(), retTypes, func_name, start, end,
                                                                ValueRange(argValues));

                    std::cerr << "------NEW CALL-------" << std::endl;

                    rewriter.setInsertionPointToEnd(out_func.getBodyBlock());
                    visitOperation(start, [&](Operation *op) {
                        auto newOp = rewriter.clone(*op);

                        for (auto val : op->getResults()) {
                            if (liveouts.find(val) != liveouts.end()) {
                                retValues[retNum[val]] = newOp->getResult(
                                        val.getResultNumber());
                                int retnum = retNum[val];
                                std::cerr << "---------REPLACE---------" << val.getResultNumber() << std::endl;
                                val.dump();
                                std::vector<OpOperand*> use_vec;
                                for (auto &use : val.getUses()) {
                                    use_vec.push_back(&use);
                                }
                                for (auto &use : use_vec) {
                                    auto bop = use->getOwner();
                                    val.replaceUsesWithIf(callOp.getResult(retnum), [&](OpOperand &operand) {
                                        return operand.getOwner() == bop;
                                    });
                                    bop->dump();
                                }
                                std::cerr << "-------------------------" << std::endl;
                            } else {
                                val.replaceAllUsesWith(newOp->getResult(val.getResultNumber()));
                            }
                        }

//                        funcOp->dump();
//                        op->dump();
//                        out_func->dump();
//                        for (auto val : op->getUsers()) {
//                            val->dump();
//                        }
//                        std::cerr << "111111111111111111111" << std::endl;
//                        std::cerr << op->use_empty();
//                        for (auto &test : op->getUses()) {
//                            test.drop();
//                        }
//                        op->dropAllUses();
//                        std::cerr << "222222222222222222222" << std::endl;
                        rewriter.eraseOp(op);
//                        std::cerr << "333333333333333333333" << std::endl;
                        std::cerr << "------ERASE OP-------" << std::endl;
                    });

                    std::cerr << "------GET RET--------" << std::endl;

//                    visitOperation(start, [&](Operation *op) {
//                        rewriter.eraseOp(op);
//                    });
                    out_func.walk([&](Operation *op) {
                        for (unsigned idx = 0; idx != op->getNumOperands(); ++idx) {
                            auto val = op->getOperand(idx);
                            auto bop = val.getDefiningOp();
                            /*if (argNum.find(val) != argNum.end()) {
                                op->setOperand(idx, out_func.getArgument(
                                        argNum[val]));
                            }*/
                            if (liveins.find(val) != liveins.end()) {
                                op->setOperand(idx, out_func.getArgument(
                                        argNum[val]));
                            }
                            continue;
                            if (!bop) {
                                if (auto arg = val.dyn_cast<BlockArgument>()) {

                                }
                                continue;
                            }
                            for (auto bval : bop->getResults()) {
                                if (bval == val) {
                                    if (liveins.find(bval) != liveins.end()) {
                                        op->setOperand(idx, out_func.getArgument(
                                                argNum[bval]));
                                    }
                                }
                            }
                        }
                    });
                    rewriter.create<tor::ReturnOp>(out_func.getLoc(), ValueRange(retValues));

                    std::cerr << "-------RETURN--------" << std::endl;

                    rewriter.setInsertionPointToStart(out_func.getBodyBlock());
                    auto new_timegraph = rewriter.create<tor::TimeGraphOp>(out_func.getLoc(), start, end);
                    rewriter.createBlock(&(new_timegraph.region()));
                    rewriter.setInsertionPointToStart(new_timegraph.getBody());
//                    rewriter.create<tor::StartTimeOp>(out_func.getLoc(), start);
                    eraseEdges.clear();
                    ++visitMarkCount;
                    markStatic(start, end);

                    std::cerr << "-------MARKED--------" << std::endl;

                    /*for (auto &edge : eraseEdges) {
                        std::cerr << edge << ";";
                        std::cerr << edge->to << ";";
                        std::cerr << std::endl;
                    }*/

                    std::vector<std::vector<Attribute>> froms(TIME_NODE);
                    std::vector<std::vector<Attribute>> attrs(TIME_NODE);
                    auto newTimeEdge = [&](TimeEdge *edge) {
                        edge->valid = false;
                        std::cerr << edge->from << ";";
                        std::cerr << edge->to << ";";
                        std::cerr << std::endl;
                        auto succ = cast<tor::SuccTimeOp>(succOp[edge->to]);
                        std::vector<Attribute> edge_array;
                        std::vector<Attribute> node_array;
                        for (size_t j = 0; j < succ.points().size(); j++) {
                            if (succ.points()[j].cast<IntegerAttr>().getInt() != edge->from) {
                                edge_array.push_back(succ.edges()[j]);
                                node_array.push_back(succ.points()[j]);
                            }
                        }
                        if (node_array.empty()) {
                            rewriter.eraseOp(succ);
                            succOp[edge->to] = NULL;
                            return;
                        }
                        succ.edgesAttr(ArrayAttr::get(getContext(), edge_array));
                        succ.pointsAttr(ArrayAttr::get(getContext(), node_array));
                    };

                    for (auto &edge : eraseEdges) {
                        froms[edge->to].push_back(IntegerAttr::get(
                                mlir::IntegerType::get(getContext(), 32, mlir::IntegerType::Signless),
                                edge->from));
                        attrs[edge->to].push_back(edge->attr);
//                        std::cerr << "!!!!" << edge->from << edge->to << std::endl;
                    }
                    for (int node = 0; node != TIME_NODE; ++node) {
                        if (node != start && nodeMark[node] == visitMarkCount) {
                            rewriter.create<tor::SuccTimeOp>(out_func.getLoc(), node,
                                                              ArrayAttr::get(getContext(), froms[node]),
                                                              ArrayAttr::get(getContext(), attrs[node]));
                        }
                    }
                    rewriter.create<tor::FinishOp>(out_func.getLoc());
//                    rewriter.create<tor::EndTimeOp>(out_func.getLoc(), end);
                    std::vector<Attribute> new_from, new_attr;
                    new_from.push_back(IntegerAttr::get(
                            mlir::IntegerType::get(getContext(), 32, mlir::IntegerType::Signless),
                            start));
                    llvm::SmallVector<NamedAttribute, 8> dict_attr;
                    dict_attr.push_back(std::make_pair(Identifier::get("type", getContext()),
                                                       StringAttr::get(getContext(), "static-call")));
                    new_attr.push_back(DictionaryAttr::get(getContext(), dict_attr));
                    rewriter.setInsertionPoint(succOp[end]);
                    auto newSuccOp = rewriter.create<tor::SuccTimeOp>(succOp[end]->getLoc(), end,
                                                                       ArrayAttr::get(getContext(), new_from),
                                                                       ArrayAttr::get(getContext(), new_attr));
                    for (auto &edge : eraseEdges) {
                        newTimeEdge(edge);
                    }
//                    rewriter.eraseOp(succOp[end]);
                    succOp[end] = newSuccOp;
                    if (pipe) {
                        out_func->setAttr("pipeline", StringAttr::get(getContext(), "for"));
                        Operation *forOp;
                        for (auto &sop : *out_func.getBodyBlock()) {
                            if (isa<tor::ForOp>(sop)) {
                                forOp = &sop;
                            }
                        }
                        out_func->setAttr("II", forOp->getAttr("II"));
                    }

                    out_func->dump();
                    funcOp->dump();
//                    exit(-1);
                };
                std::function<int(int)> getEndPoint = [&](int start) {
                    ++visitEndCount;
                    int end = -1;
                    std::function<void(int)> dfs = [&](int x) {
                        if (end != -1)
                            return;
                        if (visitEnd[x] == visitEndCount)
                            return;
                        visitEnd[x] = visitEndCount;
                        bool found = false;
                        for (auto &edge : timeGraph[x]) {
                            if (edge.Static && edge.valid) {
                                dfs(edge.to);
                                found = true;
                            }
                        }
                        if (!found && x != start) {
                            end = x;
                        }
                    };
                    dfs(start);
                    return end;
                };
                int foundVisit[TIME_NODE], foundCount = 0;
                memset(foundVisit, 0, sizeof(foundVisit));
                ++foundCount;
                std::function<void(int)> foundStaticPart = [&](int start) {
                    if (foundVisit[start] == foundCount)
                        return;
                    foundVisit[start] = foundCount;
                    int count = std::count_if(timeGraph[start].begin(), timeGraph[start].end(),
                                              [&](const TimeEdge &edge) {
                                                  return !edge.loop_exit;
                                              });
                    std::cerr << start << ":::" << count << std::endl;
                    if (count == 1) {
                        int end = start;
                        while (1) {
                            int count = std::count_if(timeGraph[end].begin(), timeGraph[end].end(),
                                                      [&](const TimeEdge &edge) {
                                                          return !edge.loop_exit;
                                                      });
                            if (count == 0) {
                                break;
                            } else if (count == 1) {
                                TimeEdge edge = timeGraph[end][0];
                                if (edge.loop_exit) {
                                    edge = timeGraph[end][1];
                                }
                                if (edge.Static && edge.valid) {
                                    end = edge.to;
                                } else {
                                    break;
                                }
                            } else {
                                assert("FOUND BRANCH not IF" && ifEnd[end] != -1);
                                if (strongConnected(end, ifEnd[end])) {
                                    end = ifEnd[end];
                                } else break;
                            }
                        }
                        if (start != end) {
                            outline(start, end);
                            foundStaticPart(end);
                        } else {
                            foundStaticPart(timeGraph[start][0].to);
                        }
                    } else if (count) {
                        int end;
                        while (end = getEndPoint(start), end != -1) {
                            outline(start, end);
                        }
                        for (auto &edge : timeGraph[start]) {
                            foundStaticPart(edge.to);
                        }
                    } else {
                        for (auto &edge : timeGraph[start]) {
                            foundStaticPart(edge.to);
                        }
                    }
                };
                if (foundDynamic) {
                    foundStaticPart(START);
                }
                for (uint32_t start = 0; start <= MAX_INDEX; ++start) {
                    for (auto &edge : timeGraph[start]) {
                        if (edge.pipeline) {
                            for_op = forOp[start];
                            outline(start, edge.to, true);
                        }
                    }
                }
                /*for (uint32_t i = 0; i < MAX_INDEX; i++) {
                  for (auto &edge : timeGraph[i]) {
                  if (!edge.Static) {
                  auto succ = cast<tor::SuccTimeOp>(succOp[edge.to]);
                  std::vector<Attribute> edge_array;
                  for (size_t j = 0; j < succ.points().size(); j++) {
                  if (succ.points()[j].cast<IntegerAttr>().getInt() == i) {
                  std::vector<NamedAttribute> dict;
                  for (auto entry : succ.edgesAttr()[j].cast<DictionaryAttr>()) {
                  if (entry.first.str() != "type") {
                  dict.push_back(entry);
                  } else {
                  dict.push_back(
                  NamedAttribute(entry.first, StringAttr::get(getContext(), "dynamic")));
                  }
                  }
                  auto new_dict = DictionaryAttr::get(getContext(), dict);
                  edge_array.push_back(new_dict);
                  } else {
                  edge_array.push_back(succ.edges()[j]);
                  }
                  }
                  auto array = ArrayAttr::get(getContext(), edge_array);
                  succ.edgesAttr(array);
                  }
                  }
                  }*/
                funcOp->getParentOp()->dump();
                return success();
            }
        };

#undef TIME_NODE
    }
    struct SplitPass : public TORSplitBase<SplitPass> {
        void runOnOperation() override {
            mlir::ModuleOp m = getOperation();
            if (m.walk([&](tor::FuncOp op) {
                        mlir::RewritePatternSet patterns(&getContext());
                        patterns.insert<split::SplitSchedule>(op.getContext());

                        if (failed(applyOpPatternsAndFold(op, std::move(patterns))))
                            return WalkResult::advance();

                        return WalkResult::advance();
                    })
                    .wasInterrupted()) {
                return signalPassFailure();
            }
        }
    };

    std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
    createTORSplitPass() {
        return std::make_unique<SplitPass>();
    }

} // namespace mlir
