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

#define DEBUG_TYPE "pipeline-parition"

namespace mlir {
    namespace partition {
#define TIME_NODE 105

        struct TimeEdge {
            int to;
            int times;
            double latency;
            bool pipeline;
            bool loop_exit;
            bool Static;
            std::vector<Operation *> ops;

            TimeEdge(int _to, int _t, bool _exit, double _lat = 0, bool _pip = false) {
                to = _to;
                times = _t;
                latency = _lat;
                pipeline = _pip;
                loop_exit = _exit;
                ops.clear();
                Static = true;
            }
        };

        double opRes[10] = {2, 2, 4, 4, 5, 6, 1, 2};

        double calculate_resource(std::vector<Operation *> &ops) {
            double resource = 0;
#define RESOURCE(Op_type, num) if(isa<Op_type>(op)) \
            resource += opRes[num];
            for (auto op : ops) {
                RESOURCE(tor::AddIOp, 0)
                RESOURCE(tor::SubIOp, 1)
                RESOURCE(tor::AddFOp, 2)
                RESOURCE(tor::SubFOp, 3)
                RESOURCE(tor::MulIOp, 4)
                RESOURCE(tor::MulFOp, 5)
                RESOURCE(tor::CmpIOp, 6)
                RESOURCE(tor::CmpFOp, 7)
            }
#undef RESOURCE
            return resource;
        }

        template<class T>
        void calculate_op(std::vector<Operation *> &ops, int &lastcnt) {
            int cnt = 0;
            for (auto op : ops) {
                if (isa<T>(op)) {
                    ++cnt;
                }
            }
            lastcnt = std::max(cnt, lastcnt);
        }

        std::vector<TimeEdge> timeGraph[TIME_NODE];
        int ifEnd[TIME_NODE];
        Operation *succOp[TIME_NODE];

        void bind_operation(int src, int dest, Operation *op) {
            for (auto &edge : timeGraph[src]) {
                if (edge.to == dest) {
                    edge.ops.push_back(op);
                }
            }
        }

        struct DP_storage {
            int start, end;
            double II;
            int type;
        };

        struct DP {
            //        double II;
            double resource;
            int start, end;
            double II;
            int type;
            std::vector<DP_storage> tmp;

            DP() {
                tmp.clear();
                type = -100;
            }

            DP(double _r, int _t = -1, int _s = -1, int _e = -1, double _i = -1) {
                resource = _r;
                type = _t;
                start = _s;
                end = _e;
                II = _i;
                tmp.clear();
            }

            friend bool operator<(const DP &x, const DP &y) {
                //            if(x.II!=y.II) {
                //                return x.II < y.II;
                //            }
                return x.resource < y.resource;
            }
        };

        std::map<std::pair<int, int>, std::map<double, DP>> dp;

        double getStaticII(int start, int end, bool *usable) {
            double II = 1;
            bool visit[TIME_NODE];
            int visit_time[TIME_NODE], new_time = 0;
            memset(visit, 0, sizeof(visit));
            memset(visit_time, 0, sizeof(visit_time));
            std::function<double(int, double, Operation *)> findOp = [&](int x, double len, Operation *op_find) {
                if (visit_time[x] == new_time)
                    return 0.0;
                visit_time[x] = new_time;
                double length = 0;
                for (auto &edge : timeGraph[x]) {
                    if (usable[edge.to]) {
                        for (auto op : edge.ops) {
                            if (op == op_find) {
                                return len;
                            }
                        }
                        length = findOp(edge.to, len + edge.latency, op_find);
                        if (length != 0)
                            return length;
                    }
                }
                return 0.0;
            };
            std::function<void(int x)> maxEdge = [&](int x) {
                if (visit[x])
                    return;
                visit[x] = true;
                for (auto &edge : timeGraph[x]) {
                    if (usable[edge.to]) {
                        II = std::max(II, edge.latency);
                        //FIXME: II for memory operation, carefully for load and store at the same cycle
                        for (auto &op : edge.ops) {
                            if (isa<tor::LoadOp, tor::StoreOp>(op)) {
                                op->dump();
                                fprintf(stderr, "=============\n");
                                for (auto val : op->getResults()) {
                                    for (auto it = val.getUses().begin(); it != val.getUses().end(); ++it) {
                                        auto bop = it.getUser();
                                        if (isa<tor::LoadOp, tor::StoreOp>(bop)) {
                                            if (!(isa<tor::LoadOp>(op) && isa<tor::LoadOp>(bop))) {
                                                ++new_time;
                                                bop->dump();
                                                II = std::max(II, findOp(x, 0, bop));
                                            }
                                        }
                                    }
                                }
                                fprintf(stderr, "#############\n");
                            }
                        }
                        maxEdge(edge.to);
                    }
                }
            };
            maxEdge(start);
            //        if (std::count_if(timeGraph[start].begin(), timeGraph[start].end(),
            //                          [&](const TimeEdge x) { return x.to <= end; }) == 1) {
            //            for (auto &edge : timeGraph[start]) {
            //                if (edge.to == end) {
            //                    return edge.latency != 0 ? edge.latency : 1;
            //                }
            //            }
            //        }
            //        return 3;
            return II;
        }

        void dynamic_programming(int start, int end) {
            if (dp.find(std::make_pair(start, end)) != dp.end())
                return;
            //        std::cerr << "BEGIN: " << start << ", " << end << std::endl;
            std::map<double, DP> mp;
            if (start == end) {
                mp[1] = DP(0);
            } else {
                bool usable[TIME_NODE];
                bool vis[TIME_NODE];
                memset(usable, 0, sizeof(usable));
                memset(vis, 0, sizeof(vis));
                std::function<bool(int)> markNode = [&](int time) {
                    if (vis[time]) {
                        return usable[time];
                    }
                    vis[time] = true;
                    if (time == end) {
                        usable[time] = true;
                        return true;
                    }
                    bool flag = false;
                    for (auto &edge : timeGraph[time]) {
                        if (markNode(edge.to) == true) {
                            usable[time] = true;
                            flag = true;
                        }
                    }
                    return flag;
                };
                markNode(start);
                //            for (int i = 0; i < 7; i++) {
                //                std::cerr << usable[i];
                //            }
                //            std::cerr << std::endl;
                int branch = std::count_if(timeGraph[start].begin(), timeGraph[start].end(),
                                           [&](const TimeEdge &x) { return usable[x.to]; });
                std::cerr << "??" << start << "," << end << "," << branch << std::endl;

                //                bool vis[TIME_NODE];
                std::function<bool(int, int)> checkModule = [&](int time, int exit) {
                    //                    std::cerr << "FOUND: " << time << ", " << exit << std::endl;
                    if (time == exit) {
                        return true;
                    }
                    bool flag = false;
                    for (auto &edge : timeGraph[time]) {
                        if (usable[edge.to]) {
                            if (!checkModule(edge.to, exit)) {
                                return false;
                            } else
                                flag = true;
                        }
                    }
                    return flag;
                };
                for (int time = 0; time < TIME_NODE; time++) {
                    if (usable[time]) {
                        if (time == start) {
                            if (branch == 1) {
                                for (auto &edge : timeGraph[time]) {
                                    if (usable[edge.to]) {
                                        dynamic_programming(edge.to, end);
                                        for (auto decision : dp[std::make_pair(edge.to, end)]) {
                                            double II = std::max(std::max(1.0, decision.first), edge.latency);
                                            double resource = decision.second.resource + calculate_resource(edge.ops);
                                            if (mp.find(II) != mp.end()) {
                                                if (DP(resource) < mp[II]) {
                                                    mp[II] = DP(resource, 1, edge.to, end, decision.first);
                                                }
                                            } else {
                                                mp[II] = DP(resource, 1, edge.to, end, decision.first);
                                            }
                                        }
                                    }
                                }
                            } else {
                                //                            std::cerr << "@@@" << time << " " << ifEnd[time] << std::endl;
                                assert(timeGraph[time].size() == 2);

                                if (ifEnd[time] == end) {
                                    for (auto &edge : timeGraph[time]) {
                                        if (usable[edge.to]) {
                                            dynamic_programming(edge.to, ifEnd[time]);
                                        }
                                    }
                                    int all_times = timeGraph[time][0].times + timeGraph[time][1].times;
                                    double p0 = timeGraph[time][0].times * 1.0 / all_times;
                                    double p1 = timeGraph[time][1].times * 1.0 / all_times;
                                    //                                std::cerr << p0 << "," << p1 << std::endl;
                                    for (auto decision0 : dp[std::make_pair(timeGraph[time][0].to, end)]) {
                                        for (auto decision1 : dp[std::make_pair(timeGraph[time][1].to, end)]) {
                                            double II =
                                                    std::max(std::max(1.0, timeGraph[time][0].latency),
                                                             decision0.first) *
                                                    p0 +
                                                    std::max(std::max(1.0, timeGraph[time][1].latency),
                                                             decision1.first) *
                                                    p1;
                                            double resource = decision0.second.resource + decision1.second.resource +
                                                              calculate_resource(timeGraph[time][0].ops) +
                                                              calculate_resource(timeGraph[time][1].ops);
                                            if (mp.find(II) != mp.end()) {
                                                if (DP() < mp[II]) {
                                                    mp[II] = DP(resource, 3, timeGraph[time][0].to, end,
                                                                decision1.first);
                                                    mp[II].tmp.push_back(
                                                            DP_storage{start, timeGraph[time][1].to, decision0.first,
                                                                       3});
                                                }
                                            } else {
                                                mp[II] = DP(resource, 3, timeGraph[time][0].to, end, decision1.first);
                                                mp[II].tmp.push_back(
                                                        DP_storage{start, timeGraph[time][1].to, decision0.first, 3});
                                            }
                                        }
                                    }
                                } else {
                                    int split = ifEnd[time];
                                    dynamic_programming(start, split);
                                    dynamic_programming(split, end);
                                    for (auto decision0 : dp[std::make_pair(start, split)]) {
                                        for (auto decision1 : dp[std::make_pair(split, end)]) {
                                            double II = std::max(decision0.first, decision1.first);
                                            if (mp.find(II) != mp.end()) {
                                                if (DP(decision0.second.resource +
                                                       decision1.second.resource) < mp[II]) {
                                                    mp[II] = DP(
                                                            decision0.second.resource +
                                                            decision1.second.resource,
                                                            4,
                                                            split, end, decision1.first);
                                                    mp[II].tmp.push_back(DP_storage{start, split, decision0.first, 4});
                                                }
                                            } else {
                                                mp[II] = DP(
                                                        decision0.second.resource +
                                                        decision1.second.resource,
                                                        4,
                                                        split, end, decision1.first);
                                                mp[II].tmp.push_back(DP_storage{start, split, decision0.first, 4});
                                            }
                                        }
                                    }
                                }
                            }
                        } else if (time != end && checkModule(start, time)) {
                            //                        std::cerr << "NEW" << start << "," << time << std::endl;
                            dynamic_programming(start, time);
                            dynamic_programming(time, end);
                            for (auto decision0 : dp[std::make_pair(start, time)]) {
                                for (auto decision1 : dp[std::make_pair(time, end)]) {
                                    double II = std::max(decision0.first, decision1.first);
                                    if (mp.find(II) != mp.end()) {
                                        if (DP(decision0.second.resource +
                                               decision1.second.resource) < mp[II]) {
                                            mp[II] = DP(
                                                    decision0.second.resource +
                                                    decision1.second.resource,
                                                    2,
                                                    time, end, decision1.first);
                                            mp[II].tmp.push_back(DP_storage{start, time, decision0.first, 2});
                                        }
                                    } else {
                                        mp[II] = DP(
                                                decision0.second.resource +
                                                decision1.second.resource,
                                                2,
                                                time, end, decision1.first);
                                        mp[II].tmp.push_back(DP_storage{start, time, decision0.first, 2});
                                    }
                                }
                            }
                        }
                    }
                }
                int opNum[10] = {};
                //double opRes[10] = {2, 2, 4, 4, 5, 6, 1, 2};
                double staticResource = 0;
                for (int i = 0; i < TIME_NODE; i++) {
                    if (usable[i]) {
                        for (auto &edge : timeGraph[i]) {
                            if (usable[edge.to]) {
                                calculate_op<tor::AddIOp>(edge.ops, opNum[0]);
                                calculate_op<tor::SubIOp>(edge.ops, opNum[1]);
                                calculate_op<tor::AddFOp>(edge.ops, opNum[2]);
                                calculate_op<tor::SubFOp>(edge.ops, opNum[3]);
                                calculate_op<tor::MulIOp>(edge.ops, opNum[4]);
                                calculate_op<tor::MulFOp>(edge.ops, opNum[5]);
                                calculate_op<tor::CmpIOp>(edge.ops, opNum[6]);
                                calculate_op<tor::CmpFOp>(edge.ops, opNum[7]);
                            }
                        }
                    }
                }
                for (int i = 0; i < 8; i++) {
                    staticResource += opNum[i] * opRes[i];
                }
                double II = getStaticII(start, end, usable);
                if (mp.find(II) != mp.end()) {
                    if (DP(staticResource) < mp[II]) {
                        mp[II] = DP(staticResource, 0);
                    }
                } else {
                    mp[II] = DP(staticResource, 0);
                }
                std::cerr << "!!!" << start << "," << end << "::" << II << std::endl;
            }
            dp[std::make_pair(start, end)] = mp;
            for (auto decision : dp[std::make_pair(start, end)]) {
                std::cerr << decision.first << " " << decision.second.resource << std::endl;
            }
            std::cerr << "----------------------------\n";
            //        for (auto decision : mp) {
            //            std::cerr << decision.first << " " << decision.second.resource << std::endl;
            //        }
        }

        void solve_pipeline(int startTime, int exitTime) {
            std::cerr << startTime << "," << exitTime << std::endl;
            std::function<int(int)> find_end = [&](int time) {
                if (timeGraph[time].empty()) {
                    return time;
                }
                for (auto &edge : timeGraph[time]) {
                    if (edge.to != exitTime) {
                        int result = find_end(edge.to);
                        if (result != -1)
                            return result;
                    }
                }
                return -1;
            };
            int endTime = find_end(startTime);
            assert(endTime != -1);
            std::cerr << endTime << std::endl;
            dp.clear();
            dynamic_programming(startTime, endTime);

            std::function<void(int, int, double)> graphRewrite = [&](int start, int end, double II) {
                if (start == -1 || end == -1) {
                    return;
                }
                std::cerr << "!!" << start << "," << end << "," << II << "," << dp[std::make_pair(start, end)][II].type
                          << std::endl;
                auto choice = dp[std::make_pair(start, end)][II];
                if (choice.type == 0) {
                    return;
                } else if (choice.type == 1) {
                    for (auto &edge : timeGraph[start]) {
                        if (edge.to == choice.start) {
                            edge.Static = false;
                        }
                    }
                    graphRewrite(choice.start, choice.end, choice.II);
                } else if (choice.type == 2) {
                    graphRewrite(choice.start, choice.end, choice.II);
                    for (auto cho : choice.tmp) {
                        graphRewrite(cho.start, cho.end, cho.II);
                    }
                } else if (choice.type == 3) {
                    for (auto &edge : timeGraph[start]) {
                        edge.Static = false;
                    }
                    graphRewrite(choice.start, choice.end, choice.II);
                    for (auto cho : choice.tmp) {
                        graphRewrite(cho.start, cho.end, cho.II);
                    }
                } else if (choice.type == 4) {
                    graphRewrite(choice.start, choice.end, choice.II);
                    for (auto cho : choice.tmp) {
                        graphRewrite(cho.start, cho.end, cho.II);
                    }
                }
            };
            //        graphRewrite(startTime, endNode, 3);
            graphRewrite(startTime, endTime, dp[std::make_pair(startTime, endTime)].begin()->first);
        }

        struct SchedulePartition : public OpRewritePattern<tor::FuncOp> {
            SchedulePartition(MLIRContext *context) : OpRewritePattern<tor::FuncOp>(context, 1) {}

            LogicalResult
            matchAndRewrite(tor::FuncOp funcOp, PatternRewriter &rewriter) const override {
                if (funcOp->hasAttr("strategy")) {
                    if (auto str = funcOp->getAttr("strategy").dyn_cast<StringAttr>()) {
                        if (str.getValue() != "mixed") {
                            funcOp->setAttr("strategy", StringAttr::get(getContext(), "mixed"));
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
                bool found = false;
                tor::TimeGraphOp timeGraphOp;
                for (auto &block : funcOp) {
                    for (auto &op : block) {
                        if (auto timegraph = dyn_cast<tor::TimeGraphOp>(op)) {
                            timeGraphOp = timegraph;
                            found = true;
                        }
                    }
                }
                if (!found) {
                    return success();
                }
                timeGraphOp->dump();
                START = timeGraphOp.starttime();
                END = timeGraphOp.endtime();
                MAX_INDEX = std::max(START, END);
                for (auto &block : timeGraphOp.region()) {
                    for (auto &op : block) {
                        if (auto start = dyn_cast<tor::StartTimeOp>(op)) {
                            MAX_INDEX = std::max(MAX_INDEX, start.starttime());
                        } else if (auto end = dyn_cast<tor::EndTimeOp>(op)) {
                            MAX_INDEX = std::max(MAX_INDEX, end.endtime());
                        } else if (auto succ = dyn_cast<tor::SuccTimeOp>(op)) {
                            //                        succ.dump();
                            //                        succ.edges().dump();
                            //                        std::cerr << succ.time() << std::endl;
                            //                        succ.points().dump();
                            MAX_INDEX = std::max(MAX_INDEX, succ.time());
                            for (unsigned i = 0; i < succ.points().size(); i++) {
                                auto from = succ.points()[i];
                                auto comp_edge = succ.edges()[i].cast<DictionaryAttr>();
                                auto edge = comp_edge.get("format");
                                int index = from.cast<IntegerAttr>().getInt();
                                auto info = edge.cast<StringAttr>().getValue().str();
                                if (info.find("static:") != StringRef::npos) {
                                    timeGraph[index].push_back(
                                            TimeEdge(succ.time(), comp_edge.get("times").cast<IntegerAttr>().getInt(),
                                                     false, std::stoi(info.substr(7))));
                                } else if (edge.cast<StringAttr>().getValue() == "static-while+pipeline") {
                                    timeGraph[index].push_back(
                                            TimeEdge(succ.time(), comp_edge.get("times").cast<IntegerAttr>().getInt(),
                                                     true, 0, true));
                                } else if (edge.cast<StringAttr>().getValue() == "static-for+pipeline") {
                                    timeGraph[index].push_back(
                                            TimeEdge(succ.time(), comp_edge.get("times").cast<IntegerAttr>().getInt(),
                                                     true, 0, true));
                                } else if (edge.cast<StringAttr>().getValue() == "static-while") {
                                    timeGraph[index].push_back(
                                            TimeEdge(succ.time(), comp_edge.get("times").cast<IntegerAttr>().getInt(),
                                                     true, 0, false));
                                } else if (edge.cast<StringAttr>().getValue() == "static-for") {
                                    timeGraph[index].push_back(
                                            TimeEdge(succ.time(), comp_edge.get("times").cast<IntegerAttr>().getInt(),
                                                     true, 0, false));
                                } else {
                                    edge.dump();
                                    assert("Unexpected edge attribute" && false);
                                }
                                std::cerr << "???" << succ.time() << std::endl;
                                succOp[succ.time()] = &op;
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
#undef BIND
                    if (auto ifOp = dyn_cast<tor::IfOp>(op)) {
                        ifEnd[ifOp.starttime()] = ifOp.endtime();
                    } else if (auto whileOp = dyn_cast<tor::WhileOp>(op)) {
                    } else if (auto forOp = dyn_cast<tor::ForOp>(op)) {
                    }
                });
                for (uint32_t i = 0; i < MAX_INDEX; i++) {
                    std::cerr << i << ": ";
                    for (auto &edge : timeGraph[i]) {
                        std::cerr << "<" << edge.to << "," << edge.times; //<< std::endl;
                        //                    for (auto sop : edge.ops) {
                        //                        std::cerr << "\t";
                        //                        sop->dump();
                        //                    }
                        std::cerr << ">; ";
                        //                    std::cerr << std::endl;
                    }
                    std::cerr << std::endl;
                }
                for (uint32_t i = 0; i < MAX_INDEX; i++) {
                    for (auto &edge : timeGraph[i]) {
                        if (edge.loop_exit && edge.pipeline) {
                            solve_pipeline(i, edge.to);
                        }
                    }
                }
                for (uint32_t i = 0; i < MAX_INDEX; i++) {
                    for (auto &edge : timeGraph[i]) {
                        if (!edge.Static) {
                            auto succ = cast<tor::SuccTimeOp>(succOp[edge.to]);
                            std::vector<Attribute> edge_array;
                            for (size_t j = 0; j < succ.points().size(); j++) {
                                if (succ.points()[j].cast<IntegerAttr>().getInt() == i) {
                                    std::vector<NamedAttribute> dict;
                                    for (auto entry : succ.edgesAttr()[j].cast<DictionaryAttr>()) {
                                        if (entry.first.str() != "format") {
                                            dict.push_back(entry);
                                        } else {
                                            dict.push_back(
                                                    NamedAttribute(entry.first,
                                                                   StringAttr::get(getContext(), "dynamic")));
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
                }
                //            exit(-1);
                return success();
            }
        };

#undef TIME_NODE
    }
    struct PipelinePartitionPass : public TORPipelinePartitionBase<PipelinePartitionPass> {
        void runOnOperation() override {
            mlir::ModuleOp m = getOperation();
            if (m.walk([&](tor::FuncOp op) {
                        mlir::RewritePatternSet patterns(&getContext());
                        patterns.insert<partition::SchedulePartition>(op.getContext());

                        if (failed(applyOpPatternsAndFold(op, std::move(patterns))))
                            return WalkResult::advance();
//                        applyOpPatternsAndFold(op, std::move(patterns));

                        return WalkResult::advance();
                    })
                    .wasInterrupted()) {
                return signalPassFailure();
            }
        }
    };

    std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
    createTORPipelinePartitionPass() {
        return std::make_unique<PipelinePartitionPass>();
    }

} // namespace mlir
