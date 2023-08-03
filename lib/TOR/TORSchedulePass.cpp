#include "TOR/PassDetail.h"
// #include "mlir/Analysis/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "TOR/Passes.h"
#include "TOR/TOR.h"
#include "TOR/TORDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
// #include "mlir/Transforms/Utils.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Schedule/SDCSchedule.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>
#define DEBUG_TYPE "create-tor"

using namespace mlir::arith;

class DisjointSet {
public:
  DisjointSet(int sz) : size(sz) {
    parent.reserve(size);
    for (int i = 0; i < size; ++i)
      parent[i] = i;
  }
  int find(int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); }
  /**
   * @brief merge x to y. ALERT: x and y are NOT interchangalbe
   * @param x
   * @param y
   */
  void merge(int x, int y) { parent[find(x)] = find(y); }

private:
  int size;
  std::vector<int> parent;
};

class TimeGraph {
private:
  struct Edge {
    std::string ds;
    int from, to;
    int length;
    int tripcount;
    int II;
  };

  int numNode;
  std::vector<std::vector<Edge>> edge;
  std::vector<std::vector<Edge>> redge;
  std::vector<std::pair<int, int>> intvMap;

public:
  TimeGraph() {
    numNode = 1;
    edge.clear();
    redge.clear();
    edge.push_back(std::vector<Edge>());
    redge.push_back(std::vector<Edge>());
  }
  int addNode() {
    edge.push_back(std::vector<Edge>());
    redge.push_back(std::vector<Edge>());
    numNode += 1;
    return numNode - 1;
  }
  void addEdge(int from, int to, std::string type, int length, int II = -1,
               int tripcount = -1) {
    edge[from].push_back(Edge{type, from, to, length, tripcount, II});
    redge[to].push_back(Edge{type, from, to, length, tripcount, II});
  }
  mlir::Attribute makeAttr(mlir::MLIRContext *ctx, Edge &edge) {
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    std::string retStr(edge.ds);

    if (edge.length > 0)
      retStr += std::string(":") + std::to_string(edge.length);
    attrs.push_back(mlir::NamedAttribute(mlir::StringAttr::get(ctx, "type"),
                                         mlir::StringAttr::get(ctx, retStr)));

    if (edge.II != -1) {
      attrs.push_back(mlir::NamedAttribute(
          mlir::StringAttr::get(ctx, "pipeline"),
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), 1)));
      attrs.push_back(mlir::NamedAttribute(
          mlir::StringAttr::get(ctx, "II"),
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), edge.II)));
    }

    if (edge.tripcount != -1)
      attrs.push_back(mlir::NamedAttribute(
          mlir::StringAttr::get(ctx, "times"),
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                 edge.tripcount)));

    mlir::DictionaryAttr dict = mlir::DictionaryAttr::get(ctx, attrs);
    return dict;
  }
  void print() {
    std::cout << "-----Here is Time Graph-----" << std::endl;
    for (int i = 0; i < numNode; i++) {
      std::cout << i << ": ";
      for (auto e : edge[i]) {
        std::cout << e.to << "(" << e.length << ") ";
      }
      std::cout << std::endl;
    }
    std::cout << "-----------------------------" << std::endl;
  }

  void rewrite(mlir::Region &region, mlir::PatternRewriter &rewriter) {
    std::vector<std::vector<mlir::Attribute>> froms(numNode);
    std::vector<std::vector<mlir::Attribute>> attrs(numNode);

    for (int i = 0; i < numNode; i++) {
      for (auto e : edge[i]) {
        froms[e.to].push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(region.getContext(), 32,
                                   mlir::IntegerType::Signless),
            e.from));
        attrs[e.to].push_back(makeAttr(region.getContext(), e));
      }
    }

    mlir::Location loc = region.getLoc();
    rewriter.setInsertionPointToStart(&region.front());
    auto timeGraphOp =
        rewriter.create<mlir::tor::TimeGraphOp>(loc, 0, numNode - 1);
    rewriter.createBlock(&timeGraphOp.getBodyRegion());
    rewriter.setInsertionPointToStart(timeGraphOp.getBody());

    // Assume start-time is 0
    for (int i = 1; i < numNode; i++) {
      rewriter.create<mlir::tor::SuccTimeOp>(
          loc, i, mlir::ArrayAttr::get(region.getContext(), froms[i]),
          mlir::ArrayAttr::get(region.getContext(), attrs[i]));
    }
    rewriter.create<mlir::tor::FinishOp>(loc);
  }

  /**
   * @brief remove the nodes whose indegs are 1 and length are 0
   */
  void canonicalize(std::vector<int> &newId) {
    DisjointSet dset(numNode);
    for (int i = 1; i < numNode; ++i) {
      if (redge[i].size() > 1)
        continue;
      auto &e = redge[i][0];
      if (e.ds == "static" && e.length == 0) {
        llvm::outs() << i << " " << e.ds << " " << e.length << "\n";
        dset.merge(i, e.from);
      }
    }

    int reducedNum = 0;
    // std::vector<int> newId(numNode, 0);
    newId.resize(numNode);

    for (int i = 0; i < numNode; ++i)
      if (dset.find(i) == i)
        newId[i] = reducedNum++;

    for (int i = 0; i < numNode; ++i)
      if (dset.find(i) != i)
        newId[i] = newId[dset.find(i)];

    llvm::outs() << numNode << " " << reducedNum << "\n";

    std::vector<std::vector<Edge>> oldedges(std::move(edge));
    std::vector<std::vector<Edge>> oldredges(std::move(redge));

    edge.resize(reducedNum);
    redge.resize(reducedNum);

    for (int i = 0; i < numNode; ++i)
      for (auto &e : oldedges[i]) {
        int u = dset.find(e.from), v = dset.find(e.to);
        if (u == v)
          continue;

        addEdge(newId[u], newId[v], e.ds, e.length, e.II, e.tripcount);
      }

    numNode = reducedNum;
  }
};

void setIntvAttr(mlir::Operation *op, std::pair<int, int> intv) {
  op->setAttr("starttime",
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 32,
                                         mlir::IntegerType::Signless),
                  intv.first));

  op->setAttr("endtime",
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 32,
                                         mlir::IntegerType::Signless),
                  intv.second));
}

int buildTimeGraphBlock(TimeGraph &tg, std::vector<mlir::Operation *> &vec,
                        int prev, scheduling::ScheduleBase *scheduler) {
  std::set<int> timeStamp;
  std::map<int, int> ts2Node;

  for (auto op : vec) {
    auto intv = scheduler->queryOp(op);
    // this op runs in [intv.fist, intv.second + 1)
    timeStamp.insert(intv.first);
    timeStamp.insert(intv.second);
  }

  int last = -1;
  for (auto ts : timeStamp) {
    int node = -1;

    if (last != -1) {
      node = tg.addNode();
      tg.addEdge(prev, node, "static", ts - last);
    } else {
      node = tg.addNode();
      tg.addEdge(prev, node, "static", 0);
    }
    ts2Node[ts] = node;
    prev = node;
    last = ts;
  }

  for (auto op : vec) {
    auto cycle = scheduler->queryOp(op);
    auto intv = std::make_pair(ts2Node[cycle.first], ts2Node[cycle.second]);
    setIntvAttr(op, intv);
  }

  vec.clear();
  return prev;
}

int buildTimeGraph(TimeGraph &tg, mlir::Block &block, int prev,
                   scheduling::ScheduleBase *scheduler) {
  int currentNode = prev;

  std::vector<mlir::Operation *> vec;

  for (auto &op : block) {
    if (auto ifOp = llvm::dyn_cast<mlir::tor::IfOp>(op)) {

      currentNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);

      int entryNode = tg.addNode();
      tg.addEdge(currentNode, entryNode, "static", 0);

      if (!ifOp.getElseRegion().empty()) {

        int thenNode = buildTimeGraph(tg, ifOp.getThenRegion().front(),
                                      entryNode, scheduler);
        int elseNode = buildTimeGraph(tg, ifOp.getElseRegion().front(),
                                      entryNode, scheduler);

        int nxtNode = tg.addNode();
        tg.addEdge(thenNode, nxtNode, "static", 0);
        tg.addEdge(elseNode, nxtNode, "static", 0);

        setIntvAttr(&op, std::make_pair(entryNode, nxtNode));
        currentNode = nxtNode;
      } else {

        int thenNode = buildTimeGraph(tg, ifOp.getThenRegion().front(),
                                      entryNode, scheduler);
        // int nxtNode = tg.addNode(thenNode, "static", 0);
        int nxtNode = thenNode;

        setIntvAttr(&op, std::make_pair(entryNode, nxtNode));
        currentNode = nxtNode;
      }

    } else if (auto whileOp = llvm::dyn_cast<mlir::tor::WhileOp>(op)) {

      currentNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);
      int beginNode = tg.addNode();
      tg.addEdge(currentNode, beginNode, "static", 0);
      int condNode =
          buildTimeGraph(tg, whileOp.getBefore().front(), beginNode, scheduler);
      int endNode = buildTimeGraph(tg, whileOp.getAfter().front(), condNode,
                                   scheduler); // body
      int nxtNode = 0;

      auto info = scheduler->queryLoop(&op);

      if (info.first == true) {
        op.setAttr("pipeline",
                   mlir::IntegerAttr::get(
                       mlir::IntegerType::get(op.getContext(), 32), 1));

        op.setAttr("II", mlir::IntegerAttr::get(
                             mlir::IntegerType::get(op.getContext(), 32),
                             info.second));
      }

      nxtNode = tg.addNode();
      tg.addEdge(beginNode, nxtNode, "static-while", 0, info.second);

      setIntvAttr(&op, std::make_pair(beginNode, endNode));
      currentNode = nxtNode;
    } else if (auto forOp = llvm::dyn_cast<mlir::tor::ForOp>(op)) {

      currentNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);
      int beginNode = tg.addNode();
      tg.addEdge(currentNode, beginNode, "static", 0);

      int endNode = buildTimeGraph(tg, *forOp.getBody(), beginNode, scheduler);
      int nxtNode = 0;

      auto info = scheduler->queryLoop(&op);

      if (info.first == true) {
        op.setAttr("pipeline",
                   mlir::IntegerAttr::get(
                       mlir::IntegerType::get(op.getContext(), 32), 1));

        op.setAttr("II", mlir::IntegerAttr::get(
                             mlir::IntegerType::get(op.getContext(), 32),
                             info.second));
      }

      nxtNode = tg.addNode();
      tg.addEdge(beginNode, nxtNode, "static-for", 0, info.second);

      setIntvAttr(&op, std::make_pair(beginNode, endNode));
      currentNode = nxtNode;
    } else {
      if (llvm::isa<mlir::tor::YieldOp>(op))
        continue;
      if (llvm::isa<mlir::tor::ConditionOp>(op))
        continue;
      if (llvm::isa<mlir::tor::ReturnOp>(op))
        continue;
      if (llvm::isa<ConstantOp>(op))
        continue;

      vec.push_back(&op);
    }
  }

  if (!vec.empty())
    currentNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);

  return currentNode;
}

mlir::LogicalResult removeExtraEdges(mlir::tor::FuncOp funcOp, TimeGraph *tg) {
  std::vector<int> newId;
  tg->canonicalize(newId);
  if (funcOp
          .walk([&](mlir::Operation *op) {
            if (op->getDialect()->getNamespace() !=
                mlir::tor::TORDialect::getDialectNamespace())
              return mlir::WalkResult::skip();
            if (auto starttime =
                    op->getAttrOfType<mlir::IntegerAttr>("starttime")) {
              auto t = starttime.getInt();
              op->setAttr("starttime",
                          mlir::IntegerAttr::get(
                              mlir::IntegerType::get(funcOp.getContext(), 32),
                              newId[t]));
            }
            if (auto endtime =
                    op->getAttrOfType<mlir::IntegerAttr>("endttime")) {
              auto t = endtime.getInt();
              op->setAttr("endtime",
                          mlir::IntegerAttr::get(
                              mlir::IntegerType::get(funcOp.getContext(), 32),
                              newId[t]));
            }

            return mlir::WalkResult::advance();
          })
          .wasInterrupted())
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult scheduleOps(mlir::tor::FuncOp funcOp,
                                mlir::PatternRewriter &rewriter) {
  using namespace scheduling;
  auto name =
      funcOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()).str();
  if (auto strategy = funcOp->getAttrOfType<StringAttr>("strategy")) {
    llvm::errs() << name << " is dynamic. No static scheduling\n";
    if (strategy.getValue().str() == "dynamic")
      return mlir::success();
  }

  std::unique_ptr<SDCSchedule> scheduler =
      std::make_unique<SDCSchedule>(SDCSchedule(funcOp.getOperation()));

  llvm::outs() << "Scheduling function " << name << "\n";

  if (mlir::succeeded(scheduler->runSchedule()))
    llvm::errs() << "Schedule Succeeded\n";
  else {
    llvm::errs() << "Schedule Failed\n";
    return mlir::failure();
  }

  TimeGraph *tg = new TimeGraph();

  buildTimeGraph(*tg, funcOp.getRegion().front(), 0, scheduler.get());

  /*
  if (failed(removeExtraEdges(funcOp, tg)))
    return mlir::failure();
  */

  tg->rewrite(funcOp.getBody(), rewriter);

  return mlir::success();
}

namespace mlir {
struct FuncOpLowering : public OpRewritePattern<mlir::tor::FuncOp> {
  using OpRewritePattern<mlir::tor::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tor::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<NamedAttribute, 4> attributes;
    for (const auto &attr : funcOp->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == mlir::function_interface_impl::getTypeAttrName())
        continue;
      attributes.push_back(attr);
    }

    llvm::SmallVector<mlir::Type, 8> argTypes;
    for (auto &arg : funcOp.getArguments()) {
      mlir::Type type = arg.getType();
      argTypes.push_back(type);
    }

    llvm::SmallVector<mlir::Type, 8> resTypes;
    for (auto &resultType : funcOp.getFunctionType().getResults()) {
      resTypes.push_back(resultType);
    }

    //   // Add control input/output to function arguments/results
    // auto noneType = rewriter.getNoneType();
    // argTypes.push_back(noneType);
    // resTypes.push_back(noneType);

    // Signature conversion (converts function arguments)
    int arg_count = funcOp.getNumArguments() + 1;
    TypeConverter::SignatureConversion result(arg_count);

    for (unsigned idx = 0, e = argTypes.size(); idx < e; ++idx)
      result.addInputs(idx, argTypes[idx]);

    // Create function of appropriate type
    auto func_type = rewriter.getFunctionType(argTypes, resTypes);
    auto newFuncOp = rewriter.create<mlir::tor::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), func_type, attributes);

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    if (failed(scheduleOps(newFuncOp, rewriter)))
      return failure();

    rewriter.eraseOp(funcOp);

    return success();
  }
};

struct TORSchedulePass : public TORScheduleBase<TORSchedulePass> {
  void runOnOperation() override {
    mlir::tor::DesignOp designOp = getOperation();

    auto result = designOp.walk([&](tor::FuncOp op) {
      // IterativeConstantFolding(op);
      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<FuncOpLowering>(designOp.getContext());
      if (failed(applyOpPatternsAndFold(op, std::move(patterns))))
        WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

std::unique_ptr<mlir::OperationPass<mlir::tor::DesignOp>>
createTORSchedulePass() {
  return std::make_unique<TORSchedulePass>();
}

} // namespace mlir
