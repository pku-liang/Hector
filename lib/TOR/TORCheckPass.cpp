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
#include <fstream>
#include <iostream>
#include <algorithm>
#include <functional>

#include "Schedule/ResourceDB.h"
#include "nlohmann/json.hpp"

#define DEBUG_TYPE "check-schedule"

namespace mlir {
    namespace check {
#define TIME_NODE 105
        scheduling::ResourceDB RDB;
        double clock_period;

        struct TimeEdge {
            int from;
            int to;
            bool loop_exit;
            bool Static;
            bool valid;
            bool pipeline;
            int latency;
            Attribute attr;
            std::vector<Operation *> ops;

            TimeEdge(int _from, int _to, Attribute _attr, bool _static = true, bool _exit = false, bool _pipe = false,
                     int _lat = 0) {
                from = _from;
                to = _to;
                attr = _attr;
                loop_exit = _exit;
                ops.clear();
                Static = _static;
                valid = true;
                pipeline = _pipe;
                latency = _lat;
            }
        };

        std::vector<TimeEdge> timeGraph[TIME_NODE];
        std::map<Operation *, int> beginNode;
        std::map<Operation *, int> endNode;

        int getWidth(Type type) {
            if (auto integer = type.dyn_cast<IntegerType>()) {
                if (integer.getWidth() == 1000) {
                    return 0;
                }
                return integer.getWidth();
            }
            if (type.isa<Float32Type>()) {
                return 32;
            }
            if (type.isa<Float64Type>()) {
                return 64;
            } else {
                type.dump();
                assert(false && "Undefined Type in TOR Dialect");
                return -1;
            }
        }

        int getLatency(Operation *op) {
            return RDB.getLatency(RDB.getResourceID(op));
        }

        double getDelay(Operation *op) {
            if (op->getNumOperands() == 0)
                return RDB.getDelay(RDB.getResourceID(op), 0);
            if (isa<tor::LoadOp>(op)) {
                return RDB.getDelay(RDB.getResourceID(op), getWidth(op->getOperand(1).getType()));
            }
            return RDB.getDelay(RDB.getResourceID(op), getWidth(op->getOperand(0).getType()));
        }

        int distance[TIME_NODE][TIME_NODE];

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
            if (isa<tor::ForOp>(op)) {
                dest = src;
            }
            beginNode[op] = src;
            endNode[op] = dest;
            if (isa<tor::ForOp, tor::WhileOp, tor::IfOp>(op)) {
                return;
            }
            for (auto &edge : timeGraph[src]) {
                if (edge.to == dest || connected(edge.to, dest)) {
                    edge.ops.push_back(op);
                }
            }
        }

        std::map<Operation *, double> delayLengths;

        double find_length(Operation *op, const std::vector<Operation *> &ops) {
            if (getLatency(op) != 0) {
                return getDelay(op);
            }
            if (delayLengths.find(op) != delayLengths.end()) {
                return delayLengths[op];
            }
            double length = getDelay(op);
            for (auto val : op->getOperands()) {
                if (!val.isa<BlockArgument>()) {
                    auto sop = val.getDefiningOp();
                    if (isa<ConstantOp, tor::AllocOp>(sop)) {
                        continue;
                    }
                    if (std::find(ops.begin(), ops.end(), sop) != ops.end()) {
                        length = std::max(length, find_length(sop, ops));
                    }
                }
            }
            delayLengths[op] = length;
            return length;
        }

        bool check_combinational(const std::vector<Operation *> &ops, double length) {
            delayLengths.clear();
            for (auto &op : ops) {
                double curLen = find_length(op, ops);
                if (curLen >= length) {
                    op->dump();
                    return false;
                } else {
                    std::cerr << "{" << curLen << "}: ";
                    op->dump();
                }
            }
            return true;
        }

        bool check(Operation *op, int time_node) {
            if (!op || isa<ConstantOp>(op) || isa<tor::AllocOp>(op))
                return true;
            if (beginNode.find(op) == beginNode.end()) {
                std::cerr << "NOT FOUND: ";
                op->dump();
                return true;
            }
            int length = distance[beginNode[op]][time_node];
            std::cerr << "[" << beginNode[op] << "->" << time_node << " ; " << length << "]:  ";
            op->dump();
            return length >= getLatency(op);
        }

        bool check_dependence(Operation *op) {
            if (!op || isa<ConstantOp>(op))
                return true;
            op->dump();

            if (beginNode.find(op) != beginNode.end()) {
                for (auto val : op->getOperands()) {
                    if (!val.isa<BlockArgument>() && !isa<ConstantOp>(val.getDefiningOp())) {
                        if (!check(val.getDefiningOp(), beginNode[op])) {
                            return false;
                        }
                    }
                }
            } else {
                std::cerr << "NOT FOUND TIME: ";
                op->dump();
            }

            for (auto &region : op->getRegions()) {
                for (auto &block : region.getBlocks()) {
                    for (auto &sop : block) {
                        if (!check_dependence(&sop)) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }

        bool check_dependence(tor::FuncOp funcOp) {
//            funcOp->dump();

            for (int i = 0; i < TIME_NODE; i++) {
                timeGraph[i].clear();
            }
            beginNode.clear();
            endNode.clear();

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
//            timeGraphOp->dump();
            START = timeGraphOp.starttime();
            END = timeGraphOp.endtime();
            MAX_INDEX = std::max(START, END);
            for (auto &block : timeGraphOp.region()) {
                for (auto &op : block) {
//                    op.dump();
                    if (auto succ = dyn_cast<tor::SuccTimeOp>(op)) {
                        MAX_INDEX = std::max(MAX_INDEX, succ.time());
                        for (unsigned i = 0; i < succ.points().size(); i++) {
                            auto from = succ.points()[i];
                            auto comp_edge = succ.edges()[i].cast<DictionaryAttr>();
                            bool pipeline = comp_edge.get("pipeline").operator bool();
                            if (pipeline) {
//                                succ->dump();
                            }
                            auto edge_info = comp_edge.get("type");
                            int index = from.cast<IntegerAttr>().getInt();
                            auto info = edge_info.cast<StringAttr>().getValue().str();
                            if (info.find("dynamic") != StringRef::npos) {
                                timeGraph[index].push_back(
                                        TimeEdge(index, succ.time(), succ.edges()[i], false,
                                                 info.find("for") != StringRef::npos ||
                                                 info.find("while") != StringRef::npos, pipeline));
                            } else if (info.find("static") != StringRef::npos) {
                                if (info.find("static:") != StringRef::npos) {
                                    timeGraph[index].push_back(
                                            TimeEdge(index, succ.time(), succ.edges()[i], true,
                                                     info.find("for") != StringRef::npos ||
                                                     info.find("while") != StringRef::npos, pipeline,
                                                     std::atoi(info.substr(info.find("static:") + 7).c_str())));
                                } else {
                                    timeGraph[index].push_back(
                                            TimeEdge(index, succ.time(), succ.edges()[i], true,
                                                     info.find("for") != StringRef::npos ||
                                                     info.find("while") != StringRef::npos, pipeline));
                                }
                            } else {
                                edge_info.dump();
                                assert("Unexpected edge_info attribute" && false);
                            }
                        }
                    }
                }
            }
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
                BIND(tor::IfOp)
                BIND(tor::WhileOp)
                BIND(tor::ForOp)
#undef BIND
            });

            memset(distance, 0x3f, sizeof(distance));
            for (int i = 1; i <= MAX_INDEX; i++) {
                distance[i][i] = 0;
                for (auto &edge : timeGraph[i]) {
                    assert(distance[i][edge.to] > 1000000000 && "Same edges");
                    distance[i][edge.to] = edge.latency;
                    if (!check_combinational(edge.ops, edge.latency * clock_period)) {
                        return false;
                    }
                }
            }
/*            for (int i = 1; i <= MAX_INDEX; i++) {
                for (int j = 1; j <= MAX_INDEX; j++) {
                    printf("%d ", distance[i][j]);
                }
                puts("");
            }*/


            for (int k = 1; k <= MAX_INDEX; k++) {
                for (int i = 1; i <= MAX_INDEX; i++) {
                    for (int j = 1; j <= MAX_INDEX; j++) {
                        distance[i][j] = std::min(distance[i][j], distance[i][k] + distance[k][j]);
                    }
                }
            }

            for (auto &op : funcOp.getRegion().front()) {
                if (isa<tor::TimeGraphOp, ConstantOp>(op)) {
                    continue;
                }
                if (!check_dependence(&op)) {
                    return false;
                }
            }


            return true;
        }

#undef TIME_NODE
    }
    struct CheckPass : public TORCheckBase<CheckPass> {
        void runOnOperation() override {
            mlir::ModuleOp m = getOperation();
            std::string filename;
            m.walk([&](tor::FuncOp op) {
                if (op.getName() == "main") {
                    check::clock_period = op->getAttrOfType<mlir::FloatAttr>("clock").getValueAsDouble();
                    if (auto attr = op->getAttrOfType<mlir::StringAttr>("resource"))
                        filename = attr.getValue().str();
                    else
                        assert(0 && "A path to the resource constraint file must be specified\n");
                }
            });
            std::ifstream istrm(filename, std::ios::in);

            nlohmann::json config;
            istrm >> config;
            check::RDB = scheduling::ResourceDB(config);


            if (m.walk([&](tor::FuncOp op) {
                        /*mlir::RewritePatternSet patterns(&getContext());
                        patterns.insert<check::CheckSchedule>(op.getContext());

                        if (failed(applyOpPatternsAndFold(op, std::move(patterns))))
                            return WalkResult::advance();

                        return WalkResult::advance();*/
                        if (!check::check_dependence(op))
                            return WalkResult::interrupt();
                        return WalkResult::advance();
                    })
                    .wasInterrupted()) {
                std::cerr << "FAILED!!!!!!\n";
                exit(-1);
                return signalPassFailure();
            }
            std::cerr << "SUCCESS!!!!!!\n";
        }
    };

    std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
    createTORCheckPass() {
        return std::make_unique<CheckPass>();
    }

} // namespace mlir
