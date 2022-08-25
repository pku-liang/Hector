#ifndef CDFG_H
#define CDFG_H

#include "TOR/TOR.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

#include <queue>
#include <string>
#include <unordered_map>

namespace scheduling {

using namespace mlir;

class Loop;
class BasicBlock;
class OpAbstract;
class ControlEdge;
class Dependence;
class OpConcrete;
class MemOpConcrete;

using ResourceType = std::string;

struct ControlEdge {
public:
  enum EdgeType { COND, FORWARD, LOOPBACK };

  ControlEdge(BasicBlock *f, BasicBlock *t, EdgeType ty)
      : fromBB(f), toBB(t), type(ty) {}

  BasicBlock *fromBB, *toBB;
  EdgeType type;
};

struct Dependence {
public:
  enum DependenceType { D_RAW, D_WAW, D_WAR, D_RAR };

  OpAbstract *SourceOp;
  OpAbstract *DestinationOp;

  int Distance;

  DependenceType type;

  Dependence(OpAbstract *SourceOp, OpAbstract *DestinationOp, int Distance,
             DependenceType type)
      : SourceOp(SourceOp), DestinationOp(DestinationOp), Distance(Distance),
        type(type) {}
};

class OpAbstract {
public:
  /// llvm rtti
  enum OpKind {
    OK_CONCRETE,
    OK_WRAPPER,
    OK_LISTWRAPPER,
    OK_SDCWRAPPER,
    OK_ENDWRAPPER
  };
  const OpKind Kind;

  OpAbstract(OpKind K) : Kind(K) {}

  OpKind getKind() const { return Kind; }

public:
  enum class OpType {
    DEFINED_OP,
    PHI_OP,    /// x = phi(a, b) in [if, while, for]
    ASSIGN_OP, /// x = y in [yield]
    INC_OP,    /// x = x + step in [for]
    LOAD_OP,   /// load
    STORE_OP
  };

  virtual MemOpConcrete *getMemOp() = 0;
  virtual OpType getType() = 0;
  virtual Operation *getOp() = 0;
  virtual int getResource() const = 0;
  virtual int getWidth() const = 0;
  virtual void setWidth(int w) = 0;
  virtual ArrayRef<Dependence *> getPred() = 0;
  virtual ArrayRef<Dependence *> getSucc() = 0;
  virtual ArrayRef<Value> getOperands() = 0;
  virtual ArrayRef<Value> getResults() = 0;
  virtual void addPred(Dependence *D) = 0;
  virtual void addSucc(Dependence *D) = 0;
  virtual void printName(llvm::raw_ostream &os) {
    if (getType() == OpType::DEFINED_OP) {
      getOp()->dump();
    } else if (getType() == OpType::PHI_OP)
      os << "PHI_OP\n";
    else if (getType() == OpType::ASSIGN_OP)
      os << "ASSIGN_OP\n";
    else if (getType() == OpType::LOAD_OP) {
      os << "LOAD_OP\n";
      getOp()->dump();
    } else if (getType() == OpType::STORE_OP) {
      os << "STORE_OP\n";
      getOp()->dump();
    }
  }
  /// Client can call this function to retrieve schedule results
  virtual int getStartTime() = 0;

  virtual void setStartTime(int T) = 0;
  virtual Loop *getParentLoop() = 0;
  virtual BasicBlock *getParentBB() = 0;
};

class OpConcrete : public OpAbstract {
public:
  OpConcrete(Operation *op, Loop *ParentLoop, BasicBlock *ParentBB,
             int resource, ArrayRef<Value> R, ArrayRef<Value> O, OpType type)
      : OpAbstract(OK_CONCRETE), type(type), op(op), ResourceId(resource),
        ParentLoop(ParentLoop), ParentBB(ParentBB) {
    bitwidth = 0;

    Operands.clear();
    Results.clear();

    Results.insert(Results.begin(), R.begin(), R.end());
    Operands.insert(Operands.begin(), O.begin(), O.end());
  }

  int getWidth() const override { return bitwidth; }

  void setWidth(int w) override { bitwidth = w; }

  virtual MemOpConcrete *getMemOp() override { return nullptr; }

  OpType getType() override { return type; }

  Operation *getOp() override { return op; }

  int getResource() const override { return ResourceId; }

  ArrayRef<Dependence *> getPred() override { return pred; }

  ArrayRef<Dependence *> getSucc() override { return succ; }

  ArrayRef<Value> getOperands() override { return Operands; }

  ArrayRef<Value> getResults() override { return Results; }

  void addPred(Dependence *D) override { pred.push_back(D); }

  void addSucc(Dependence *D) override { succ.push_back(D); }

  int getStartTime() override { return startTime; }

  void setStartTime(int T) override { startTime = T; }

  Loop *getParentLoop() override { return ParentLoop; }

  BasicBlock *getParentBB() override { return ParentBB; }

protected:
  OpType type;
  Operation *op;
  int ResourceId;
  int bitwidth;
  SmallVector<Dependence *, 4> succ, pred;
  SmallVector<Value, 4> Operands;
  SmallVector<Value, 4> Results;
  Loop *ParentLoop;
  BasicBlock *ParentBB;
  int startTime;
};

class MemOpConcrete : public OpConcrete {
public:
  virtual MemOpConcrete *getMemOp() override { return this; }

  Value getAddr() { return addr; }

  Value getMemRef() { return memref; }

  int getPartitionIndicies() {
    assert(hasFixedMemoryBank());

    tor::MemRefType type = memref.getType().dyn_cast<tor::MemRefType>();

    auto shape = type.getShape();
    auto property = type.getProperty();

    int idx = 0;

    for (auto x : llvm::enumerate(partitionIndices)) {
      APInt attr;
      matchPattern(x.value(), m_ConstantInt(&attr));
      if (property[x.index()].getValue() == "complete")
        idx = idx * shape[x.index()] + attr.getLimitedValue();
    }
    return idx;
  }

  bool hasFixedMemoryBank() {
    SmallVector<APInt, 4> idx;
    for (auto x : partitionIndices) {
      APInt attr;
      if (!matchPattern(x, m_ConstantInt(&attr)))
        return false;
    }
    return true;
  }

  SmallVector<APInt, 4> getMemoryBankIdx() {
    assert(hasFixedMemoryBank());
    SmallVector<APInt, 4> idx;
    for (auto x : partitionIndices) {
      APInt attr;
      matchPattern(x, m_ConstantInt(&attr));
      idx.push_back(attr);
    }
    return idx;
  }

  MemOpConcrete(Operation *op, Loop *ParentLoop, BasicBlock *ParentBB,
                int resource, ArrayRef<Value> R, ArrayRef<Value> O, OpType type)
      : OpConcrete(op, ParentLoop, ParentBB, resource, R, O, type) {
    assert(llvm::isa<tor::LoadOp>(op) || llvm::isa<tor::StoreOp>(op));

    if (auto loadOp = llvm::dyn_cast<tor::LoadOp>(op)) {
      memref = loadOp.getMemRef();
      addr = loadOp.getOperand(
          1); // the first index is the address in the memory bank
      partitionIndices = loadOp.getIndices().drop_front(1);
    } else if (auto storeOp = llvm::dyn_cast<tor::StoreOp>(op)) {
      memref = storeOp.getMemRef();
      addr = storeOp.getOperand(
          1); // the first index is the address in the memory bank
      partitionIndices = storeOp.getIndices().drop_front(1);
    }
    llvm::outs() << partitionIndices.size() << "\n";
  }

  bool isMemOpConcrete() {
    return type == OpType::LOAD_OP || type == OpType::STORE_OP;
  }

private:
  Value memref;
  Value addr;
  SmallVector<Value, 4> partitionIndices;
};

/// Other scheduling algorithm may inherit this class to add extra
class OpWrapperBase : public OpAbstract {
public:
  OpWrapperBase(OpAbstract *op) : OpAbstract(OK_WRAPPER) {}

  OpWrapperBase(OpAbstract *op, OpKind K) : OpAbstract(K), op(op) {}

  virtual MemOpConcrete *getMemOp() override { return op->getMemOp(); }

  virtual int getWidth() const override { return op->getWidth(); }

  virtual void setWidth(int w) override { return op->setWidth(w); }

  virtual OpType getType() override { return op->getType(); }

  virtual Operation *getOp() override { return op->getOp(); }

  virtual int getResource() const override { return op->getResource(); }

  virtual ArrayRef<Dependence *> getPred() override { return op->getPred(); }

  virtual ArrayRef<Dependence *> getSucc() override { return op->getSucc(); }

  virtual ArrayRef<Value> getOperands() override { return op->getOperands(); }

  virtual ArrayRef<Value> getResults() override { return op->getResults(); }

  virtual void addPred(Dependence *D) override { op->addPred(D); }

  virtual void addSucc(Dependence *D) override { op->addSucc(D); }

  virtual int getStartTime() override { return op->getStartTime(); }

  virtual void setStartTime(int T) override { op->setStartTime(T); }

  virtual Loop *getParentLoop() override { return op->getParentLoop(); }

  virtual BasicBlock *getParentBB() override { return op->getParentBB(); }

protected:
  OpAbstract *op;
};

class BasicBlock {
public:
  void addOperation(OpAbstract *op) { Operations.push_back(op); }

  ArrayRef<ControlEdge> getPred() { return Pred; }

  llvm::MutableArrayRef<OpAbstract *> getOperations() { return Operations; }

  ArrayRef<ControlEdge> getSucc() { return Succ; }

  static void addControlDependency(ControlEdge edge) {
    edge.fromBB->Succ.push_back(edge);
    edge.toBB->Pred.push_back(edge);
  }

  Loop *getParentLoop() const { return ParentLoop; }

  OpAbstract *getBranchOp() const { return BranchOp; }

  void setBranchOp(OpAbstract *op) { BranchOp = op; }

  void setParentLoop(Loop *P) { ParentLoop = P; }

  void setBranchValue(Value v) { BranchValue = v; }

  Value getBranchValue() const { return BranchValue; }

  BasicBlock(Loop *ParentLoop = nullptr) : ParentLoop(ParentLoop) {
    Operations.clear();
    Pred.clear();
    Succ.clear();
    BranchOp = nullptr;
  }

private:
  llvm::SmallVector<OpAbstract *> Operations;
  llvm::SmallVector<ControlEdge, 4> Pred;
  llvm::SmallVector<ControlEdge, 4> Succ;
  Loop *ParentLoop;
  OpAbstract *BranchOp;
  Value BranchValue;
};

class Loop {
public:
  struct LoopBound {
  public:
    Value lowerBound, upperBound, stride;
  };

  Loop(Loop *ParentLoop, Operation *DefiningOp, bool PipelineFlag = false,
       int TargetII = 1)
      : PipelineFlag(PipelineFlag), TargetII(TargetII), AchievedII(-1),
        ParentLoop(ParentLoop), DefiningOp(DefiningOp) {
    LoopBody.clear();
    ChildLoop.clear();
    TopLevelBlock.clear();
  }

  ArrayRef<BasicBlock *> getBody() { return LoopBody; }

  ArrayRef<Loop *> getChildLoop() { return ChildLoop; }

  ArrayRef<BasicBlock *> getTopLevelBlock() { return TopLevelBlock; }

  Loop *getParentLoop() { return ParentLoop; }

  void addBasicBlock(BasicBlock *BB) {
    TopLevelBlock.push_back(BB);
    LoopBody.push_back(BB);
  }

  void addChildLoop(Loop *L) {
    ChildLoop.push_back(L);
    LoopBody.insert(LoopBody.end(), L->getBody().begin(), L->getBody().end());
  }

public:
  bool PipelineFlag;
  int TargetII;
  int AchievedII;

private:
  llvm::SmallVector<BasicBlock *>
      LoopBody; /// all basic blocks in this loop body
  llvm::SmallVector<BasicBlock *>
      TopLevelBlock; /// top-level basic block in this loop body
  llvm::SmallVector<Loop *> ChildLoop; /// Top-level sub loops in this loop;
  Loop *ParentLoop;
  Operation *DefiningOp;
};

/// A BFS algorithm to determine wheter operation st and reach operation en
/// without taking loop back edge. Special case: st and en are in the same block
bool canReach(OpAbstract *st, OpAbstract *en, bool backFlag);

/// Chech whether two memory operation in DAG might have memory conflict
bool hasMemPortConflict(OpAbstract *PrevMemOp, OpAbstract *LatMemOp);

/// Check whehter two memory operation in the inner-most loop can have
/// memory port conflict with a distance Dist.
bool hasMemPortConflict(OpAbstract *PrevMemOp, OpAbstract *LatMemOp, int Dist);

} // namespace scheduling

#endif
