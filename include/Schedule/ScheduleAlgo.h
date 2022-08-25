#ifndef SCHEDULE_ALGO_H
#define SCHEDULE_ALGO_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/ArrayRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Identifier.h"
#include "TOR/TOR.h"
#include "TOR/TORTypes.h"
#include "Schedule/CDFG.h"
#include "Schedule/ResourceDB.h"

#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <memory>

namespace scheduling{

using std::unique_ptr;
using namespace mlir;

/// This is a template Base class of Schedule Algorithm. Every implementation of 
/// schedule algorithm should inherit from this class. template parameter must be 
/// a child class of OpWrapperBase.
/// This class contains some basic constraints about the scheduling problem.
/// Can access schedule result from OpAbstract
/// Can access OpAbstract via result value from ValueMap
class ScheduleBase {
public:
  explicit ScheduleBase(Operation *op) {
    containingOp = op;
    EntryBB = nullptr;
    ExitBB = nullptr;

    auto funcOp = llvm::dyn_cast<tor::FuncOp>(op);
    assert(funcOp);

    if (auto attr = funcOp->getAttrOfType<mlir::FloatAttr>("clock"))
      ClockPeriod = attr.getValue().convertToDouble();
    else if (auto attr = funcOp->getAttrOfType<mlir::IntegerAttr>("clock"))
      ClockPeriod = (double)attr.getValue().roundToDouble();
    else
      assert(0 && "A clock period must be specified");

    std::string filename;    
    if (auto attr = funcOp->getAttrOfType<mlir::StringAttr>("resource"))
      filename = attr.getValue().str();
    else
      assert(0 && "A path to the resource constraint file must be specified\n");

    std::ifstream istrm(filename, std::ios::in);
    
    nlohmann::json config;
    istrm >> config;
    RDB = ResourceDB(config);
  }

  virtual LogicalResult runSchedule() = 0;

  /// verify resource constraints and dependencies
  virtual LogicalResult verify();

  virtual void buildFromContaingOp();
  
  OpAbstract *createOp(Operation *op, Loop *ParentLoop, BasicBlock *ParentBB, 
                       ArrayRef<Value> Results, ArrayRef<Value> Operands, 
                       OpAbstract::OpType type = OpAbstract::OpType::DEFINED_OP) 
  {
    int rsc = 0;
    switch (type) {
      case OpAbstract::OpType::PHI_OP:
      case OpAbstract::OpType::ASSIGN_OP:
        rsc = RDB.getResourceID("nop");
        break;
      default:
        rsc = RDB.getResourceID(op);
    }

    Operations.push_back(
        std::make_unique<OpConcrete>(
            OpConcrete(op, ParentLoop, ParentBB, rsc, Results, Operands, type)));

    OpAbstract *newop = Operations.back().get();

    int width = 0;

    if (Results.size() > 0) {
      // TODO bit width analysis
      if (Results[0].getType().isIntOrFloat())
        width = Results[0].getType().getIntOrFloatBitWidth();
      else
        width = 32;
    }

    newop->setWidth(width);

    return newop;
  }
  
  OpAbstract *createMemOp(Operation *op, Loop *ParentLoop, BasicBlock *ParentBB,
                          ArrayRef<Value> Results, ArrayRef<Value> Operands,
                          OpAbstract::OpType type)
  {
    int rsc = RDB.getResourceID("memport");
    Operations.push_back(
        std::make_unique<MemOpConcrete>(
            MemOpConcrete(op, ParentLoop, ParentBB, rsc, Results, Operands, type)));
    
    int width = 0;
    if (Results.size() > 0) {
      if (Results[0].getType().isIntOrFloat())
        width = Results[0].getType().getIntOrFloatBitWidth();
      else
        width = 32;
    }
    
    OpAbstract *newop = Operations.back().get();
    newop->setWidth(width);

    return newop;
  }

  void addDependency(Dependence D) {
    Dependencies.push_back(std::make_unique<Dependence>(D));
    D.SourceOp->addSucc(Dependencies.back().get());
    D.DestinationOp->addPred(Dependencies.back().get());
  }
  
  /// build Dependency from containingOp;
  void setClockFrequence(int Cycle) {
    ClockPeriod = Cycle;
  }

  void printCDFG();

  void printSchedule();

  /**
   * Query the scheduling information of a loop operation 
   * @param op reference to the loop operation
   * @return first: pipeline flag, second: achieved II
   */
  std::pair<int, int> queryLoop(Operation *op) {
    Loop *L = LoopMap[op];
    return std::make_pair(L->PipelineFlag, L->AchievedII);
  }

  /**
   * Query the scheduling information of a calculate operations
   * @param op address of the operation. address must remains unchanged from
   *           that of input
   * @return first: starting cycle, seoncd: ending cycle
   */
  std::pair<int, int> queryOp(Operation *op) {
    if (op->getNumResults() == 0) {
      if (OperationMap.find(op) == OperationMap.end()) 
        return std::make_pair(0, 0);
      OpAbstract *opA = OperationMap[op];
      int duration = std::max(1, RDB.getLatency(opA->getResource()));
      int start = opA->getStartTime();
      int end = start + duration;
      return std::make_pair(start, end);
    }

    int start = std::numeric_limits<int>().max(), end = 0;
    for (int i = 0, n = op->getNumResults(); i < n; ++i) {
      Value result = op->getResult(i);
      OpAbstract *opA = ValueMap[result];

      int duration = std::max(1, RDB.getLatency(opA->getResource()));
      start = std::min(start, opA->getStartTime());
      end = std::max(end, start + duration);
    }
    return std::make_pair(start, end);
  }
private:

  /**
   * @brief walk through a mlir block. 
   * @return (beginning BB, exiting BB)
   */
  std::pair<BasicBlock*, BasicBlock*> buildCFG(Block &block, Loop *ParentLoop);

  void buildDFG(); 
protected:

  /**
   * @brief convert OpAbstract to T
   */
  template<typename T>
  std::vector<std::unique_ptr<T>> initSchedule() {
    
    std::vector<std::unique_ptr<T>> vec;
    llvm::DenseMap<const OpAbstract*, T*> OpMap;

    for (auto&& x : Operations) {
      vec.push_back(std::make_unique<T>(x.get()));

      OpMap[x.get()] = vec.back().get();
    }

    for (auto&& BB : BasicBlocks) {
      for (auto& x : BB->getOperations())
        x = OpMap[x];
      BB->setBranchOp(OpMap[BB->getBranchOp()]);
    }

    for (auto&& D : Dependencies) {
      D->SourceOp = OpMap[D->SourceOp];
      D->DestinationOp = OpMap[D->DestinationOp];
    }

    for (auto &x : ValueMap)
      x.second = OpMap[x.second];

    for (auto &x : OperationMap)
      x.second = OpMap[x.second];

    return std::move(vec);
  }

protected:

  Operation *containingOp; /// This is the containing Op that needs to be scheduled. i.e. a moduleOp or a while loop Op

  float ClockPeriod;

  ResourceDB RDB;

  BasicBlock *EntryBB, *ExitBB;

  std::vector<unique_ptr<Dependence>> Dependencies; /// This vector contains all the dependencies.

  std::vector<unique_ptr<OpConcrete>> Operations; /// This vector contains all the operations that need to be scheduled.

  std::vector<unique_ptr<BasicBlock>> BasicBlocks;

  std::vector<unique_ptr<Loop>> Loops;

  std::unordered_map<Operation*, OpAbstract*> OperationMap; // Map mlir operation to OpAbstract

  llvm::DenseMap<Value, OpAbstract*> ValueMap; /// real Operation that needs to be scheduled

  llvm::DenseMap<Operation*, Loop*> LoopMap;
};

} // namespace scheduling


#endif
