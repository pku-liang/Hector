#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/ScopedHashTable.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "TOR/TORDialect.h"
#include "TOR/Passes.h"
#include "TOR/PassDetail.h"

#include <list>
#include <unordered_set>

#define DEBUG_TYPE "width-analysis"

namespace {
using namespace mlir;

APInt max(const APInt &a, const APInt &b) {
  return a.sgt(b) ? a : b;
}

APInt min(const APInt &a, const APInt &b) {
  return a.slt(b) ? a : b;
}

class RangeOp;
struct DataRange;

struct RangeInfo {
  enum RangeType {
    Invalid,
    Normal
  } type;

  APInt lb, ub;

  RangeInfo() {}
  RangeInfo(const APInt &l, const APInt &u) {
    if (l.slt(APInt(128, std::numeric_limits<int64_t>::min(), 1)))
      lb = APInt(128, std::numeric_limits<int64_t>::min(), 1);
    else
      lb = l;
    
    if (u.sgt(APInt(128, std::numeric_limits<int64_t>::max(), 1)))
      ub = APInt(128, std::numeric_limits<int64_t>::max(), 1);
    else
      ub = u;

    if (lb.sgt(ub)) type = Invalid;
    else type = Normal;
  }

  int getWidthNeeded() {
    static int64_t const limit[7] = {0, 1, 7 ,127, 32767, 2147483647, 9223372036854775807LL};
    if (lb.sge(0) && ub.sle(1)) // special case for i1
      return 1;
    for (int i = 1; i < 7; ++i)
      if (lb.sge(-limit[i] - 1) && ub.sle(limit[i]))
        return 1 << i;
    assert(0 && "Range Exceeded!");
    return -1;
  }

  static RangeInfo getUnknown() {
    return RangeInfo(APInt(128, std::numeric_limits<int64_t>::min(), 1), 
        APInt(128, std::numeric_limits<int64_t>::max(), 1));
  }

  static RangeInfo getConst(int64_t c) {
    return RangeInfo(APInt(128, c, 1), APInt(128, c, 1));
  }
  
  static RangeInfo getIntv(int64_t l, int64_t r) {
    return RangeInfo(APInt(128, l, 1), APInt(128, r, 1));
  }

  RangeInfo rev() {
    return RangeInfo(ub, lb);
  }

  RangeInfo operator + (const RangeInfo &rhs) const {
    return RangeInfo(this->lb + rhs.lb, this->ub + rhs.ub);
  }

  RangeInfo operator - (const RangeInfo &rhs) const {
    return RangeInfo(this->lb - rhs.lb, this->ub - rhs.ub);
  }

  bool operator == (const RangeInfo &rhs) const {
    return this->lb == rhs.lb && this->ub == rhs.ub;
  }

  bool operator != (const RangeInfo &rhs) const {
    return !(*this == rhs);
  }
  
  void print(raw_ostream &OS) {
    OS << "[";
    lb.print(OS, true);
    OS << ", ";
    ub.print(OS, true);
    OS << "]";
  }

  friend RangeInfo intersection_r(const RangeInfo &a, const RangeInfo &b) {
    if (a.type == Invalid || b.type == Invalid)
      return RangeInfo(APInt(128, 1), APInt(128, 0));
    return RangeInfo(max(a.lb, b.lb), min(a.ub, b.ub));
  }

  friend RangeInfo union_r(const RangeInfo &a, const RangeInfo &b) {
    if (a.type == Invalid)
      return b;
    if (b.type == Invalid)
      return a;
    return RangeInfo(min(a.lb, b.lb), max(a.ub, b.ub));
  }

  bool intersectionWith(const RangeInfo &info) {
    if (type == Invalid)
      return false;

    bool flag = lb.slt(info.lb) || ub.sgt(info.ub);
    lb = max(lb, info.lb);
    ub = min(ub, info.ub);
    
    if (lb.sgt(ub))
      type = Invalid;

    return flag;
  }
  
  bool unionWith(const RangeInfo &info) {
    if (type == Invalid)
      return false;
    if (info.type == Invalid)
      return false;

    bool flag = lb.sgt(info.lb) || ub.slt(info.ub);
    lb = min(lb, info.lb);
    ub = max(ub, info.ub);
    return flag;
  }
};

struct DataRange {
  Value value; // SSA variable this data range linking to
  RangeOp *defOp; // Defining op that results this data Rangex
  SmallVector<RangeOp*, 4> users;
  RangeInfo rangeUp;
  RangeInfo rangeDown;
  DataRange(Value value, RangeOp *defOp) : value(value), defOp(defOp) {
    rangeUp = RangeInfo::getUnknown();
    rangeDown = RangeInfo::getUnknown();
    users.clear();
  }
};

struct RangeOp {
  enum Type {
    TOR_OP,
    RangeUnion,
    RangeInt,
    RangePhi,
    RangeCondAssign, // no back propagate
  };

  RangeOp(Type type, Operation *op) : type(type), op(op) {
    results.clear();
    operands.clear();
  }

  static std::unique_ptr<RangeOp> getCondAssign() {
    return std::make_unique<RangeOp>(RangeCondAssign, nullptr);
  }

  static std::unique_ptr<RangeOp> getFromOp(Operation *op) {
    return std::make_unique<RangeOp>(TOR_OP, op);
  }

  static std::unique_ptr<RangeOp> getRangeUnion() {
    return std::make_unique<RangeOp>(RangeUnion, nullptr);
  }

  static std::unique_ptr<RangeOp> getRangeInt() {
    return std::make_unique<RangeOp>(RangeInt, nullptr);
  }

  static std::unique_ptr<RangeOp> getRangePhi() {
    return std::make_unique<RangeOp>(RangePhi, nullptr);
  }

  void addOperand(DataRange *opr) {
    assert(opr != nullptr && "error nullptr");
    operands.push_back(opr);
    opr->users.push_back(this);
  }
  Type type;
  Operation *op;
  SmallVector<DataRange*, 2> results;
  SmallVector<DataRange*, 2> operands;
};


template<typename K, typename V>
class ScopedMap {
private:
  using MapT = DenseMap<K, V>;
  std::list<MapT> Maps;
public:

  void insert(const K &key, const V &val) {
    Maps.back().insert(std::make_pair(key, val));
  }

  void insertScope() {
    Maps.push_back(MapT());
  }

  void removeCurrentScope() {
    Maps.pop_back();
  }

  V lookup(const K &key) {
    for (auto &Map : llvm::reverse(Maps))
      if (Map.find(key) != Map.end())
        return Map[key];
    return V();
  }
};

struct PredicateInfo {
public:
  SmallVector<DataRange*, 4> TrueRange;
  SmallVector<DataRange*, 4> FalseRange;
  PredicateInfo() {
    TrueRange.clear();
    FalseRange.clear();
  }
};

class RangeAnalysis {
public:
  using ValueScopedMapT = ScopedMap<Value, DataRange*>;
  using ValueMapT = DenseMap<Value, DataRange*>;
  using PredicateMap = DenseMap<Value, PredicateInfo>;

  RangeAnalysis(Operation *op) : topOperation(op) {
    build(op);
  }

  RangeInfo queryRange(Value v) {
    return ValueMap[v]->rangeDown;
  }

private:

  PredicateMap branchRangeMap;
  Operation *topOperation;
  ValueMapT ValueMap;
  std::unordered_set<RangeOp*> worklist;
  std::vector<std::unique_ptr<DataRange>> Ranges;
  std::vector<std::unique_ptr<RangeOp>> RangeOps;

  DataRange *allocRange(Value v, RangeOp *op = nullptr) {
    Ranges.push_back(std::make_unique<DataRange>(v, op));
    if (op != nullptr)
      op->results.push_back(Ranges.back().get());
    return Ranges.back().get();
  }

  void buildUpdateGraph(Block *block, ValueScopedMapT &ValueToRange) {
    auto IntegerPred = [&] (Value v) {return v.getType().isa<IntegerType>();};

    for (auto &op : block->getOperations()) {
      // TODO complex predicate involving and, or. No tor.and/or
      if (auto ifOp = llvm::dyn_cast<tor::IfOp>(op)) {

        auto &info = branchRangeMap[ifOp.getOperand()];

        // then region
        ValueToRange.insertScope();
        for (auto *r : info.TrueRange)
          ValueToRange.insert(r->value, r);

        buildUpdateGraph(&ifOp.thenRegion().front(), ValueToRange);

        SmallVector<DataRange*, 4> thenYieldOprs;
        if (ifOp.thenRegion().front().getTerminator()) {
          auto thenYield = llvm::dyn_cast<tor::YieldOp>(ifOp.thenRegion().front().getTerminator());
          for (auto opr : thenYield.getOperands())
            thenYieldOprs.push_back(ValueToRange.lookup(opr));
        }

        ValueToRange.removeCurrentScope();
        
        // else region
        if (!ifOp.elseRegion().empty()) {
          ValueToRange.insertScope();

          for (auto *r : info.FalseRange)
            ValueToRange.insert(r->value, r);

          buildUpdateGraph(&ifOp.elseRegion().front(), ValueToRange);

          SmallVector<DataRange*, 4> elseYieldOprs;
          if (ifOp.elseRegion().front().getTerminator()) {
            auto elseYield = llvm::dyn_cast<tor::YieldOp>(ifOp.elseRegion().front().getTerminator());
            for (auto opr : elseYield.getOperands())
              elseYieldOprs.push_back(ValueToRange.lookup(opr));
          }

          ValueToRange.removeCurrentScope();
          for (auto result : llvm::enumerate(ifOp.getResults())) {
            if (!IntegerPred(result.value()))
              continue;
            
            RangeOps.push_back(RangeOp::getRangePhi());
            auto curOp = RangeOps.back().get();
            auto resRange = allocRange(result.value(), curOp);

            ValueToRange.insert(result.value(), resRange);
            ValueMap.insert(std::make_pair(result.value(), resRange));

            curOp->addOperand(thenYieldOprs[result.index()]);
            curOp->addOperand(elseYieldOprs[result.index()]);
          }
        } // otherwise no need to deal with yield op
      } else if (auto whileOp = llvm::dyn_cast<tor::WhileOp>(op)) {

        for (auto arg : llvm::enumerate(whileOp.before().front().getArguments())) {
          if (!IntegerPred(arg.value()))
            continue;
          
          RangeOps.push_back(RangeOp::getRangePhi());
          RangeOp *whilePhi = RangeOps.back().get();

          DataRange *argRange = allocRange(arg.value(), whilePhi);

          ValueToRange.insert(arg.value(), argRange);
          ValueMap.insert(std::make_pair(arg.value(), argRange));
          whilePhi->addOperand(ValueToRange.lookup(whileOp.getOperand(arg.index())));
        }

        buildUpdateGraph(&whileOp.before().front(), ValueToRange);

        auto condOp = llvm::dyn_cast<tor::ConditionOp>(whileOp.before().front().getTerminator());

        for (auto res : llvm::enumerate(whileOp.getResults())) {
          if (!IntegerPred(res.value()))
            continue;
          
          RangeOps.push_back(RangeOp::getCondAssign()); 
          RangeOp *whileAssign = RangeOps.back().get();
          DataRange *resRange = allocRange(res.value(), whileAssign);
          ValueToRange.insert(res.value(), resRange);
          ValueMap.insert(std::make_pair(res.value(), resRange));
          whileAssign->addOperand(ValueToRange.lookup(condOp.getOperand(res.index() + 1)));
        }

        for (auto arg : llvm::enumerate(whileOp.before().front().getArguments())) {
          if (!IntegerPred(arg.value()))
            continue;

          RangeOps.push_back(RangeOp::getCondAssign()); 
          RangeOp *whileAssign = RangeOps.back().get();
          DataRange *argRange = allocRange(arg.value(), whileAssign);
          ValueToRange.insert(arg.value(), argRange);
          ValueMap.insert(std::make_pair(arg.value(), argRange));
          whileAssign->addOperand(ValueToRange.lookup(condOp.getOperand(arg.index() + 1)));
        }

        buildUpdateGraph(&whileOp.after().front(), ValueToRange);
        auto yieldOp = llvm::dyn_cast<tor::YieldOp>(whileOp.after().front().getTerminator());

        for (auto arg : llvm::enumerate(whileOp.before().front().getArguments())) {
          if (!IntegerPred(arg.value()))
            continue;
          
          RangeOp *whilePhi = ValueMap[arg.value()]->defOp;
          DataRange *oprRange = ValueToRange.lookup(yieldOp.getOperand(arg.index()));
          whilePhi->addOperand(oprRange);
        }

      } else if (auto forOp = llvm::dyn_cast<tor::ForOp>(op)) {
        
        RangeOps.push_back(RangeOp::getRangePhi()); // TODO this may be pessimistic
        auto forRangeOp = RangeOps.back().get();

        DataRange *iterRange = allocRange(forOp.getInductionVar(), forRangeOp);

        ValueToRange.insert(forOp.getInductionVar(), iterRange);
        ValueMap.insert(std::make_pair(forOp.getInductionVar(), iterRange));
        forRangeOp->addOperand(ValueToRange.lookup(forOp.lowerBound()));
        forRangeOp->addOperand(ValueToRange.lookup(forOp.upperBound()));

        // add iteration variabls        
        
        ArrayRef<OpOperand> initOperands = forOp.getIterOpOperands();
        for (auto iterArg : llvm::enumerate(forOp.getRegionIterArgs())) {
          if (!IntegerPred(iterArg.value()))
            continue;

          RangeOps.push_back(RangeOp::getRangePhi());
          auto curOp = RangeOps.back().get();
          auto iterArgRange = allocRange(iterArg.value(), curOp);
          ValueToRange.insert(iterArg.value(), iterArgRange);
          ValueMap.insert(std::make_pair(iterArg.value(), iterArgRange));
          curOp->addOperand(ValueToRange.lookup(initOperands[iterArg.index()].get()));
        }

        buildUpdateGraph(forOp.getBody(), ValueToRange);

        auto yieldOp = llvm::dyn_cast<tor::YieldOp>(forOp.getBody()->getTerminator());
        for (auto iterArg : llvm::enumerate(forOp.getRegionIterArgs())) {
          if (!IntegerPred(iterArg.value()))
            continue;
          
          auto iterArgRange = ValueMap[iterArg.value()];
          auto yieldOpr = ValueToRange.lookup(yieldOp.getOperand(iterArg.index()));
          auto curOp = iterArgRange->defOp;
          
          curOp->addOperand(yieldOpr);
        }

      } else if (auto cmpIOp = llvm::dyn_cast<tor::CmpIOp>(op)) {
        auto result = cmpIOp.getResult();

        PredicateInfo cmpInfo;
        RangeOps.push_back(RangeOp::getFromOp(&op));
        auto curOp = RangeOps.back().get();

        for (auto opr : cmpIOp.getOperands()) {
          auto trueRange = allocRange(opr, curOp);
          auto falseRange = allocRange(opr, curOp);

          curOp->addOperand(ValueToRange.lookup(opr));
          cmpInfo.TrueRange.push_back(trueRange);
          cmpInfo.FalseRange.push_back(falseRange);
        }

        branchRangeMap.insert(std::make_pair(result, cmpInfo));
      } else if (auto condOp = llvm::dyn_cast<tor::ConditionOp>(op)) {
        continue;
      } else {
        RangeOps.push_back(RangeOp::getFromOp(&op));
        auto curOp = RangeOps.back().get();

        SmallVector<DataRange*, 4> operands;
        for (auto opr : op.getOperands()) 
          if (IntegerPred(opr))
            operands.push_back(ValueToRange.lookup(opr));

        if (op.getNumResults() == 1 && IntegerPred(op.getResult(0))) {
          auto result = allocRange(op.getResult(0), curOp);
          ValueToRange.insert(op.getResult(0), result);
          ValueMap.insert(std::make_pair(op.getResult(0), result));
        }

        for (auto opr : operands) 
          curOp->addOperand(opr);
      }
    }
  }

  void initRanges() {
    
    for (auto &&rangeOp : RangeOps) {
      if (rangeOp->type == RangeOp::TOR_OP) {
        if (auto constOp = llvm::dyn_cast<ConstantOp>(rangeOp->op)) {
          // range of result is fixed
          if (constOp.getValue().isa<FloatAttr>())
            continue;
          DataRange *result = rangeOp->results[0];
          result->rangeDown = RangeInfo::getConst(constOp.getValue().cast<IntegerAttr>().getInt());
          result->rangeUp = RangeInfo::getConst(constOp.getValue().cast<IntegerAttr>().getInt());
          for (auto user : result->users)
            worklist.insert(user);
        } else if (auto loadOp = llvm::dyn_cast<tor::LoadOp>(rangeOp->op)) {
          auto memrefTy = loadOp.getMemRefType();
          
          // range of index can be computed
          auto shape = memrefTy.getShape();
          for (auto idx : llvm::enumerate(loadOp.getIndices())) {
            if (shape[idx.index()] == -1) // dynamic shape
              continue;
            DataRange *oprRange = rangeOp->operands[idx.index()];
            oprRange->rangeUp.intersectionWith(RangeInfo::getIntv(0, shape[idx.index()] - 1));
            worklist.insert(oprRange->defOp);
          }

          // range of loaded value can be computed
          if (!memrefTy.getElementType().isa<IntegerType>())
            continue;
          
          int width = memrefTy.getElementTypeBitWidth();
          DataRange *result = rangeOp->results[0];
          result->rangeDown = RangeInfo::getIntv(-(1LL << width), (1LL << width) - 1);
          for (auto user : result->users)
            worklist.insert(user);
        } else if (auto storeOp = llvm::dyn_cast<tor::StoreOp>(rangeOp->op)) {

          auto memrefTy = storeOp.getMemRefType();
          
          auto shape = memrefTy.getShape();
          int offset = memrefTy.getElementType().isa<IntegerType>();
          for (auto idx : llvm::enumerate(storeOp.getIndices())) {
            if (shape[idx.index()] == -1) // dynamic shape
              continue;
            DataRange *oprRange = rangeOp->operands[idx.index() + offset];
            oprRange->rangeUp.intersectionWith(RangeInfo::getIntv(0, shape[idx.index()] - 1));
            worklist.insert(oprRange->defOp);
          }

          // range of stored value can be computed
          if (memrefTy.getElementType().isa<FloatType>())
            continue;
          
          int width = memrefTy.getElementTypeBitWidth();
          DataRange *operand0 = rangeOp->operands[0];
          operand0->rangeUp = RangeInfo::getIntv((1LL << (width - 1)), (1LL << (width - 1)) - 1);
          worklist.insert(operand0->defOp);
        }
      }
    }
  }

  /**
   * @brief Update the down range of varRange by newRange.
   *        Add its users into worklist if range is changed
   * @param varRange
   * @param newRange
   */
  void updateDown(DataRange *varRange, RangeInfo newRange) {
    if (varRange->rangeDown != newRange) {
      varRange->rangeDown = newRange;
      for (auto user : varRange->users)
        worklist.insert(user);
    }
  }

  /**
   * @brief Update the down range of varRange by newRange.
   *        Add its users into worklist if range is changed
   * @param varRange
   * @param newRange
   */
  void updateUp(DataRange *varRange, RangeInfo newRange) {
    if (varRange->rangeUp != newRange) {
      varRange->rangeUp = newRange;
      if (varRange->defOp != nullptr)
        worklist.insert(varRange->defOp);
    }
  }
  
  void updateAtAddIOp(RangeOp *op) {
    // a = b + c
    DataRange *aRange = op->results[0];
    DataRange *bRange = op->operands[0];
    DataRange *cRange = op->operands[1];
    
    updateDown(aRange, intersection_r(aRange->rangeUp, bRange->rangeDown + cRange->rangeDown));
    updateUp(bRange, intersection_r(bRange->rangeDown, aRange->rangeUp - cRange->rangeDown.rev()));
    updateUp(cRange, intersection_r(cRange->rangeDown, aRange->rangeUp - bRange->rangeDown.rev()));
  }

  void updateAtSubIOp(RangeOp *op) {
    // a = b - c
    DataRange *aRange = op->results[0];
    DataRange *bRange = op->operands[0];
    DataRange *cRange = op->operands[1];
    
    updateDown(aRange, intersection_r(aRange->rangeUp, bRange->rangeDown - cRange->rangeDown.rev()));
    updateUp(bRange, intersection_r(bRange->rangeDown, aRange->rangeUp + cRange->rangeDown));
    updateUp(cRange, intersection_r(cRange->rangeDown, bRange->rangeDown - aRange->rangeUp));
  }

  void updateAtMulIOp(RangeOp *op) {
    // a = b * c
    DataRange *aRange = op->results[0];
    DataRange *bRange = op->operands[0];
    DataRange *cRange = op->operands[1];
    
    auto &b = bRange->rangeDown, &c = cRange->rangeDown;
    APInt points[4] = {b.lb * c.lb, b.lb * c.ub, b.ub * c.lb, b.ub * c.ub};
    RangeInfo mulRange(min(min(points[0], points[1]), min(points[2], points[3])),
        max(max(points[0], points[1]), max(points[2], points[3])));

    updateDown(aRange, intersection_r(aRange->rangeUp, mulRange));
    // TODO reverse update
  }

  void updateAtCmpIOp(RangeOp *op) {
    // x cmp y
    DataRange *xRange = op->operands[0];
    DataRange *yRange = op->operands[1];
    DataRange *xtrueRange = op->results[0];
    DataRange *xfalseRange = op->results[1];
    DataRange *ytrueRange = op->results[2];
    DataRange *yfalseRange = op->results[3];

    updateUp(xRange, intersection_r(xRange->rangeDown, union_r(xtrueRange->rangeUp, xfalseRange->rangeUp)));
    updateUp(yRange, intersection_r(yRange->rangeDown, union_r(ytrueRange->rangeUp, yfalseRange->rangeUp)));

    auto cmpIOp = llvm::dyn_cast<tor::CmpIOp>(op->op);
    switch (cmpIOp.predicate()) {
      case tor::CmpIPredicate::sgt:
        // x > y
        updateDown(ytrueRange, intersection_r(ytrueRange->rangeUp, 
            intersection_r(yRange->rangeDown, RangeInfo(yRange->rangeDown.lb, xRange->rangeDown.ub - 1))));
        updateDown(yfalseRange, intersection_r(yfalseRange->rangeUp, 
            intersection_r(yRange->rangeDown, RangeInfo(xRange->rangeUp.lb, yRange->rangeDown.ub))));
        updateDown(xtrueRange, intersection_r(xtrueRange->rangeUp, 
            intersection_r(xRange->rangeDown, RangeInfo(yRange->rangeDown.lb + 1, xRange->rangeDown.ub))));
        updateDown(xfalseRange, intersection_r(xfalseRange->rangeUp, 
            intersection_r(xRange->rangeDown, RangeInfo(xRange->rangeDown.lb, yRange->rangeUp.ub))));
        break;

      case tor::CmpIPredicate::sge:
        // x >= y
        updateDown(ytrueRange, intersection_r(ytrueRange->rangeUp, 
            intersection_r(yRange->rangeDown, RangeInfo(yRange->rangeDown.lb, xRange->rangeDown.ub))));
        updateDown(yfalseRange, intersection_r(yfalseRange->rangeUp, 
            intersection_r(yRange->rangeDown, RangeInfo(xRange->rangeUp.lb + 1, yRange->rangeDown.ub))));
        updateDown(xtrueRange, intersection_r(xtrueRange->rangeUp, 
            intersection_r(xRange->rangeDown, RangeInfo(yRange->rangeDown.lb, xRange->rangeDown.ub))));
        updateDown(xfalseRange, intersection_r(xfalseRange->rangeUp, 
            intersection_r(xRange->rangeDown, RangeInfo(xRange->rangeDown.lb, yRange->rangeUp.ub - 1))));
        break;

      case tor::CmpIPredicate::slt:
        // x < y
        updateDown(xtrueRange, intersection_r(xtrueRange->rangeUp, 
            intersection_r(xRange->rangeDown, RangeInfo(xRange->rangeDown.lb, yRange->rangeDown.ub - 1))));
        updateDown(xfalseRange, intersection_r(xfalseRange->rangeUp, 
            intersection_r(xRange->rangeDown, RangeInfo(yRange->rangeUp.lb, xRange->rangeDown.ub))));
        updateDown(ytrueRange, intersection_r(ytrueRange->rangeUp, 
            intersection_r(yRange->rangeDown, RangeInfo(xRange->rangeDown.lb + 1, yRange->rangeDown.ub))));
        updateDown(yfalseRange, intersection_r(yfalseRange->rangeUp, 
            intersection_r(yRange->rangeDown, RangeInfo(yRange->rangeDown.lb, xRange->rangeUp.ub))));
        break;
      case tor::CmpIPredicate::sle:
        // x <= y
        updateDown(xtrueRange, intersection_r(xtrueRange->rangeUp, 
            intersection_r(xRange->rangeDown, RangeInfo(xRange->rangeDown.lb, yRange->rangeDown.ub))));
        updateDown(xfalseRange, intersection_r(xfalseRange->rangeUp, 
            intersection_r(xRange->rangeDown, RangeInfo(yRange->rangeUp.lb + 1, xRange->rangeDown.ub))));
        updateDown(ytrueRange, intersection_r(ytrueRange->rangeUp, 
            intersection_r(yRange->rangeDown, RangeInfo(xRange->rangeDown.lb, yRange->rangeDown.ub))));
        updateDown(yfalseRange, intersection_r(yfalseRange->rangeUp, 
            intersection_r(yRange->rangeDown, RangeInfo(yRange->rangeDown.lb, xRange->rangeUp.ub - 1))));
      default:
        break;
    }
  }

  void updateAtPhi(RangeOp *op) {
    // a = phi(b, c)
    DataRange *aRange = op->results[0];
    DataRange *bRange = op->operands[0];
    DataRange *cRange = op->operands[1];

    updateDown(aRange, intersection_r(aRange->rangeUp, union_r(bRange->rangeDown, cRange->rangeDown)));
    updateUp(bRange, intersection_r(bRange->rangeDown, aRange->rangeUp));
    updateUp(cRange, intersection_r(cRange->rangeDown, aRange->rangeUp));
  }

  void updateAtCondAssign(RangeOp *op) {
    // Forward-propagation only
    // a = b [at a some condition which is unknown]
    DataRange *aRange = op->results[0];
    DataRange *bRange = op->results[1];

    updateDown(aRange, bRange->rangeDown);
  }

  void updateAtOp(RangeOp *op) {
    switch (op->type) {
      case RangeOp::TOR_OP:
        if (llvm::isa<tor::AddIOp>(op->op))
          updateAtAddIOp(op);
        else if (llvm::isa<tor::SubIOp>(op->op))
          updateAtSubIOp(op);
        else if (llvm::isa<tor::MulIOp>(op->op))
          updateAtMulIOp(op);
        else if (llvm::isa<tor::CmpIOp>(op->op))
          updateAtCmpIOp(op);
        break;
      case RangeOp::RangePhi:
        updateAtPhi(op);
        break;
      case RangeOp::RangeCondAssign:
        updateAtCondAssign(op);
      case RangeOp::RangeInt:
        // TODO
        break;
      case RangeOp::RangeUnion:
        // TODO
        break;
      default:
        break;
    }
  }

  void propagateRange() {
    worklist.clear();
    initRanges();

    while (!worklist.empty()) {
      auto op = *worklist.begin();
      worklist.erase(worklist.begin());
      updateAtOp(op);
    }
  }
  
  void addConstantOps(Operation *op, ValueScopedMapT &ValueToRange) {
    auto designOp = op->getParentOfType<tor::DesignOp>();
    assert(designOp && "Must included in a tor.design");
    designOp.walk(
      [&] (ConstantOp constOp) {
        if (constOp->getAttr("value").isa<FloatAttr>())
          WalkResult::skip();
        RangeOps.push_back(RangeOp::getFromOp(constOp.getOperation()));
        auto *curOp = RangeOps.back().get();
        DataRange *constRange = allocRange(constOp.getResult(), curOp);
        ValueToRange.insert(constOp.getResult(), constRange);
      }
    );
  }

  void build(Operation *op) {
    assert(llvm::isa<tor::FuncOp>(op) && "Must operate on a tor::FuncOp");
    
    auto funcOp = llvm::cast<tor::FuncOp>(op);

    ValueScopedMapT ValueToRange;

    ValueToRange.insertScope();

    addConstantOps(op, ValueToRange);

    for (auto arg : funcOp.getArguments()) {
      DataRange *argRange = allocRange(arg, nullptr);
      ValueMap.insert(std::make_pair(arg, argRange));
      ValueToRange.insert(arg, argRange);
    }

    buildUpdateGraph(&funcOp.body().front(), ValueToRange);
    ValueToRange.removeCurrentScope();

    propagateRange();
  }
};

void reduceWidth(tor::FuncOp funcOp) {
  RangeAnalysis *analysis = new RangeAnalysis(funcOp.getOperation());

  auto updateType = [&] (Value v) {
    if (auto ty = v.getType().dyn_cast<IntegerType>()) {
      int originalWidth = ty.getWidth();
      int newWidth = analysis->queryRange(v).getWidthNeeded();
      if (newWidth < originalWidth)
        v.setType(IntegerType::get(funcOp.getContext(), newWidth));
    }
  };

  funcOp.walk(
    [&] (Operation *op) {
      if (auto forOp = llvm::dyn_cast<tor::ForOp>(op)) {
        // change block arguments
        for (auto arg : forOp.getBody()->getArguments())
          updateType(arg);
      } else if (auto whileOp = llvm::dyn_cast<tor::WhileOp>(op)) {
        for (auto arg : whileOp.before().front().getArguments())
          updateType(arg);
        for (auto arg : whileOp.after().front().getArguments())
          updateType(arg);
      }

      for (auto result : op->getResults())
        if (auto ty = result.getType().dyn_cast<IntegerType>()) {
          int originalWidth = ty.getWidth();
          int newWidth = analysis->queryRange(result).getWidthNeeded();
          if (newWidth < originalWidth)
            result.setType(IntegerType::get(funcOp.getContext(), newWidth));
        }
      
      WalkResult::advance();
    }
  );
}

struct FuncOpAnalysisPattern : public OpRewritePattern<tor::FuncOp> {
  using OpRewritePattern<tor::FuncOp>::OpRewritePattern;


  LogicalResult
  matchAndRewrite(tor::FuncOp op, PatternRewriter &rewriter) const override 
  {
    if (op->getAttr("bitwidth-reduced"))
      return failure();
    rewriter.updateRootInPlace(op, 
    [&] {
      reduceWidth(op);
      op->setAttr("bitwidth-reduced", IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
    });
    return success();
  }
};

struct WidthAnalysisPass : public WidthAnalysisBase<WidthAnalysisPass> {
  void runOnOperation() override {
    tor::DesignOp designOp = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.insert<FuncOpAnalysisPattern>(&getContext());

    if (designOp.walk(
      [&] (tor::FuncOp op) {
        if (failed(applyOpPatternsAndFold(op, std::move(patterns))))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
    ).wasInterrupted())
      signalPassFailure();
    
    designOp.walk(
      [&] (ConstantOp constOp) {
        if (!constOp.getValue().isa<IntegerAttr>())
          return WalkResult::skip();
        int64_t val = constOp.getValue().cast<IntegerAttr>().getInt();
        RangeInfo info = RangeInfo::getConst(val);

        int width = info.getWidthNeeded();
        constOp.getResult().setType(IntegerType::get(&getContext(), width));
        constOp->setAttr("value", IntegerAttr::get(IntegerType::get(&getContext(), width), val));
        return WalkResult::advance();
      }
    );
  }
};

} // namespace

namespace mlir {

std::unique_ptr<OperationPass<tor::DesignOp>>
createWidthAnalysisPass() {
  return std::make_unique<WidthAnalysisPass>();
}

} // namespace mlir