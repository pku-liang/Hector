#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>

#include "TOR/TOR.h"
#include "Schedule/CDFG.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"

#include "Schedule/ScheduleAlgo.h"

namespace scheduling {

using namespace mlir;

/// @return std::pair<BasicBlock*, BasicBlock*> **first** is the entry BB, **second** is the exit BB
std::pair<BasicBlock*, BasicBlock*> ScheduleBase::buildCFG(Block &block, Loop *parentLoop) {

  BasicBlocks.push_back(
      std::make_unique<BasicBlock>(BasicBlock(parentLoop)));

  BasicBlock *lastHead = BasicBlocks.back().get();

  BasicBlock *exitBlock = lastHead;

  if (parentLoop != nullptr)
    parentLoop->addBasicBlock(lastHead);
  
  for (auto &op : llvm::reverse(block)) {
    if (auto ifOp = llvm::dyn_cast<tor::IfOp>(op)) {

      if (!ifOp.elseRegion().empty()) {
        
        auto thenBB = buildCFG(ifOp.thenRegion().front(), parentLoop);
        auto elseBB = buildCFG(ifOp.elseRegion().front(), parentLoop);
        
        BasicBlock::addControlDependency({thenBB.second, lastHead, ControlEdge::FORWARD});
        BasicBlock::addControlDependency({elseBB.second, lastHead, ControlEdge::FORWARD});

        auto thenYieldOp = ifOp.thenRegion().back().getTerminator();
        auto elseYieldOp = ifOp.elseRegion().back().getTerminator();

        for (unsigned i = 0, n = ifOp.getNumResults(); i < n; ++i) {
          Value x = ifOp.getResult(i);
          OpAbstract *newOpA = createOp(&op, parentLoop, lastHead, 
                                        {x},
                                        {thenYieldOp->getOperand(i), elseYieldOp->getOperand(i), 
                                          ifOp.getOperand() /* control dependency */ },
                                        OpAbstract::OpType::PHI_OP);
          ValueMap.insert(std::make_pair(x, newOpA));
          lastHead->addOperation(newOpA);
        }

        BasicBlocks.push_back(
            std::make_unique<BasicBlock>(BasicBlock(parentLoop)));

        lastHead = BasicBlocks.back().get();

        if (parentLoop != nullptr) 
          parentLoop->addBasicBlock(lastHead);

        BasicBlock::addControlDependency({lastHead, thenBB.first, ControlEdge::COND});
        BasicBlock::addControlDependency({lastHead, elseBB.first, ControlEdge::COND});
      } else {

        auto thenBB = buildCFG(ifOp.thenRegion().front(), parentLoop);
        
        BasicBlocks.push_back(
            std::make_unique<BasicBlock>(BasicBlock(parentLoop)));

        BasicBlock *condHead = BasicBlocks.back().get();

        if (parentLoop != nullptr)
          parentLoop->addBasicBlock(condHead);
        
        BasicBlock::addControlDependency({thenBB.second, lastHead, ControlEdge::FORWARD});
        BasicBlock::addControlDependency({condHead, lastHead, ControlEdge::COND});
        BasicBlock::addControlDependency({condHead, thenBB.first, ControlEdge::COND});

        lastHead = condHead;
      }

      lastHead->setBranchValue(ifOp.getOperand());

    } else if (auto whileOp = llvm::dyn_cast<tor::WhileOp>(op)) {
      bool pipelineFlag = false;
      int targetII = -1;
      
      if (auto attr = op.getAttrOfType<mlir::IntegerAttr>("pipeline")) 
        pipelineFlag = attr.getInt();
      if (auto attr = op.getAttrOfType<mlir::IntegerAttr>("II"))
        targetII = attr.getInt();
      op.removeAttr("pipeline");
      op.removeAttr("II");
      
      Loops.push_back(std::make_unique<Loop>(Loop(parentLoop, &op, pipelineFlag, targetII)));

      Loop *curLoop = Loops.back().get();
      if (parentLoop != nullptr)
        parentLoop->addChildLoop(curLoop);

      LoopMap[&op] = curLoop;

      auto bodyBB = buildCFG(whileOp.after().front(), curLoop);
      auto condBB = buildCFG(whileOp.before().front(), curLoop);

      BasicBlock::addControlDependency({condBB.second, lastHead, ControlEdge::COND});
      BasicBlock::addControlDependency({condBB.second, bodyBB.first, ControlEdge::COND});
      BasicBlock::addControlDependency({bodyBB.second, condBB.first, ControlEdge::LOOPBACK});
      
      auto yieldOp = whileOp.after().back().getTerminator();

      for (unsigned i = 0, n = whileOp.before().getNumArguments(); i < n; ++i) {
        /// phi in cond argument
        Value x = whileOp.before().getArgument(i);

        OpAbstract *newOpA = createOp(&op, curLoop, condBB.first, 
                                      {x},
                                      {whileOp.getOperand(i), yieldOp->getOperand(i)},
                                      OpAbstract::OpType::PHI_OP); // PHI happens inside the loop

        ValueMap.insert(std::make_pair(x, newOpA));
        condBB.first->addOperation(newOpA);
      }
      
      auto condOp = whileOp.before().back().getTerminator(); /// the first operand of condop is condition
      
      condBB.second->setBranchValue(condOp->getOperand(0));

      for (unsigned i = 0, n = whileOp.after().getNumArguments(); i < n; ++i) {
        /// assign in body arguments
        Value x = whileOp.after().getArgument(i);
        OpAbstract *newOpA = createOp(&op, curLoop, bodyBB.first, 
                                      {x},
                                      {condOp->getOperand(i + 1)},
                                      OpAbstract::OpType::ASSIGN_OP); // assign happens inside the loop
        ValueMap.insert(std::make_pair(x, newOpA));
        bodyBB.first->addOperation(newOpA);
      }

      for (unsigned i = 0, n = whileOp.getNumResults(); i < n; ++i) {
        /// assign in results
        Value x = whileOp.getResult(i);
        OpAbstract *newOpA = createOp(&op, parentLoop, lastHead, 
                                      {x},
                                      {condOp->getOperand(i + 1)},
                                      OpAbstract::OpType::ASSIGN_OP);
        ValueMap.insert(std::make_pair(x, newOpA));
        lastHead->addOperation(newOpA);
      }

      
      BasicBlocks.push_back(
          std::make_unique<BasicBlock>(BasicBlock(parentLoop)));

      lastHead = BasicBlocks.back().get();

      BasicBlock::addControlDependency({lastHead, condBB.first, ControlEdge::FORWARD});

    } else if (auto allocOp = llvm::dyn_cast<tor::AllocOp>(op)) {
      /// omited

    } else if (auto loadOp = llvm::dyn_cast<tor::LoadOp>(op)) {

      Value result = loadOp.getResult();
      OpAbstract *newOpA = createMemOp(&op, parentLoop, lastHead, 
                                    std::vector<Value>{result},
                                    std::vector<Value>{loadOp.getIndices().begin(), loadOp.getIndices().end()},
                                    OpAbstract::OpType::LOAD_OP);
      ValueMap[result] = newOpA;
      OperationMap[&op] = newOpA;
      lastHead->addOperation(newOpA);

    } else if (auto storeOp = llvm::dyn_cast<tor::StoreOp>(op)) {

      Value mem = storeOp.getMemRef();
      std::vector<Value> operands{storeOp.getIndices().begin(),
				  storeOp.getIndices().end()
      };
      operands.push_back(storeOp.getValueToStore());
      
      OpAbstract *newOpA = createMemOp(&op, parentLoop, lastHead,
				       std::vector<Value>{mem},
				       operands,
				       OpAbstract::OpType::STORE_OP);
      OperationMap[&op] = newOpA;
      lastHead->addOperation(newOpA);
    } else if (auto forOp = llvm::dyn_cast<tor::ForOp>(op)) {

      bool pipelineFlag = false;
      int targetII = -1;
      
      if (auto attr = op.getAttrOfType<mlir::IntegerAttr>("pipeline")) 
        pipelineFlag = attr.getInt();
      if (auto attr = op.getAttrOfType<mlir::IntegerAttr>("II"))
        targetII = attr.getInt();
      
      op.removeAttr("pipeline");
      op.removeAttr("II");
      
      Loops.push_back(std::make_unique<Loop>(Loop(parentLoop, &op, pipelineFlag, targetII)));

      Loop *curLoop = Loops.back().get();
      if (parentLoop != nullptr)
        parentLoop->addChildLoop(curLoop);

      LoopMap[&op] = curLoop;

      auto bodyBB = buildCFG(*forOp.getBody(), curLoop);
      BasicBlock::addControlDependency({bodyBB.second, bodyBB.first, ControlEdge::LOOPBACK});
      BasicBlock::addControlDependency({bodyBB.first, lastHead, ControlEdge::COND});

      auto yieldOp = forOp.getBody()->getTerminator();

      Value induction = forOp.getInductionVar();
      OpAbstract *newOpA = createOp(&op, curLoop, bodyBB.first, 
                                    std::vector<Value>{induction},
                                    std::vector<Value>{forOp.lowerBound(), induction},
                                    OpAbstract::OpType::PHI_OP);
      bodyBB.first->addOperation(newOpA);
      ValueMap.insert(std::make_pair(induction, newOpA));

      // forOp iteration variables
      for (auto iter : llvm::enumerate(forOp.getRegionIterArgs())) {
        Value x = iter.value();
        OpAbstract *newOpA = createOp(&op, curLoop, bodyBB.first,
                                      std::vector<Value>{x},
                                      std::vector<Value>{forOp.initArgs()[iter.index()], yieldOp->getOperand(iter.index())},
                                      OpAbstract::OpType::PHI_OP);
        ValueMap.insert(std::make_pair(x, newOpA));
        bodyBB.first->addOperation(newOpA);
      }

      // forOp results
      for (auto result : llvm::enumerate(forOp.getResults())) {
        Value x = result.value();
        OpAbstract *newOpA = createOp(&op, parentLoop, lastHead, 
                                      {x},
                                      {yieldOp->getOperand(result.index())},
                                      OpAbstract::OpType::ASSIGN_OP);
        ValueMap.insert(std::make_pair(x, newOpA));
        lastHead->addOperation(newOpA);
      }

      BasicBlocks.push_back(std::make_unique<BasicBlock>(BasicBlock(parentLoop)));
      lastHead = BasicBlocks.back().get();
      BasicBlock::addControlDependency(ControlEdge(lastHead, bodyBB.first, ControlEdge::FORWARD));

    } else if (auto condOp = llvm::dyn_cast<tor::ConditionOp>(op)) {
      /// omited
      continue;
    } else if (auto yieldOp = llvm::dyn_cast<tor::YieldOp>(op)) {
      /// omited
      continue;
    } else {
      OpAbstract *newOpA = createOp(&op, parentLoop, lastHead, 
                                    std::vector<Value>{op.getResults().begin(), op.getResults().end()},
                                    std::vector<Value>{op.getOperands().begin(), op.getOperands().end()});
      lastHead->addOperation(newOpA);
      OperationMap[&op] = newOpA;
      for (auto result : op.getResults()) {
        Value x = result;
        ValueMap.insert(std::make_pair(x, newOpA));
      }
    }
  }

  return std::make_pair(lastHead, exitBlock);
}

void ScheduleBase::buildDFG() {
  // dependence of scalars
  for (auto&& BB : BasicBlocks) {
    for (auto op : BB->getOperations()) {
      if (op->getType() == OpAbstract::OpType::PHI_OP) {
        if (llvm::isa<tor::WhileOp>(op->getOp()) || llvm::isa<tor::ForOp>(op->getOp())) {
          // second operand is loop recursion dependence
          addDependency(Dependence(ValueMap[op->getOperands()[0]], op, 0, Dependence::D_RAW));
          addDependency(Dependence(ValueMap[op->getOperands()[1]], op, 1, Dependence::D_RAW));
          continue;
        } 
        // otherwise is phi in an ifop
      }

      for (auto v : op->getOperands())
        addDependency(Dependence(ValueMap[v], op, 0, Dependence::D_RAW));
    }   
  }

  // dependence of tensors
  for (auto&& op1 : Operations) {
    if (op1->getType() != OpAbstract::OpType::LOAD_OP &&
        op1->getType() != OpAbstract::OpType::STORE_OP)
      continue;

    for (auto&& op2 : Operations) {
      if (op1.get() == op2.get())
        continue;

      if (op2->getType() != OpAbstract::OpType::LOAD_OP &&
          op2->getType() != OpAbstract::OpType::STORE_OP)
        continue;

      auto memop1 = op1->getMemOp();
      auto memop2 = op2->getMemOp();

      if (memop1->getMemRef() != memop2->getMemRef())
        // no dependency
        continue;

      // check if memop1 and memop2 use different memory bank 
      if (memop1->hasFixedMemoryBank() && memop2->hasFixedMemoryBank()) {
        if (memop1->getPartitionIndicies() != memop2->getPartitionIndicies())
          continue;
      }
      
      // check if the effect of memop1 while reach memop2
      int Distance = -1;

      // memop1 can reach memop2 without loop-back edge
      if (canReach(memop1, memop2, false))
        Distance = 0;

      // memop1 can reach memop2 using some loop-back edge,
      // pessimiticaly assume distance to be 1
      if (Distance == -1 && canReach(memop1, memop2, true))
        Distance = 1;

      // first check if memop1 can reach memop2 without taking loop-back edge
      // if (canReachBB(memop1->getParentBB(),
      //                memop2->getParentBB(),
      //                [&](ControlEdge succ){return succ.type !=
      //                ControlEdge::LOOPBACK;}))
      //   Distance = 0;
      // // check if memop1 can reach memop2 using loop-back edge. pessimiticaly
      // set the distance to 1 if (Distance == -1 &&
      // canReachBB(memop1->getParentBB(), memop2->getParentBB()))
      //   Distance = 1;

      // memop1 can't reach memop2 (e.g. mutually exclusive branch of an if statement)
      if (Distance == -1)
        continue;
      
      if (op1->getType() == OpAbstract::OpType::LOAD_OP &&
          op2->getType() == OpAbstract::OpType::LOAD_OP) 
      {
        // No data dependency
        continue;
      }
      
      if (op1->getType() == OpAbstract::OpType::LOAD_OP &&
          op2->getType() == OpAbstract::OpType::STORE_OP)
        // WAR
        addDependency(Dependence(memop1, memop2, Distance, Dependence::D_WAR));
	
      else if (op1->getType() == OpAbstract::OpType::STORE_OP &&
          op2->getType() == OpAbstract::OpType::LOAD_OP)
        // RAW
        addDependency(Dependence(memop1, memop2, Distance, Dependence::D_RAW));

      else if (op1->getType() == OpAbstract::OpType::STORE_OP &&
          op2->getType() == OpAbstract::OpType::STORE_OP)
        // WAW
        addDependency(Dependence(memop1, memop2, Distance, Dependence::D_WAW));
    }
  }
}

void ScheduleBase::buildFromContaingOp() {
  if (auto funcOp = llvm::dyn_cast<tor::FuncOp>(containingOp)) {
    /// instantiate a funOp
    Region &region = funcOp.getBody();
    auto bbs = buildCFG(region.front(), nullptr);

    for (auto&& BB : BasicBlocks)
      if (BB->getBranchValue().getImpl() != nullptr) {
        BB->setBranchOp(ValueMap[BB->getBranchValue()]);
        assert(ValueMap[BB->getBranchValue()]);
      }

    /// add function op at beginning
    OpAbstract *funcArgOp = 
        createOp(containingOp, nullptr, bbs.first, 
                 {funcOp.getArguments().begin(), funcOp.getArguments().end()}, {}, 
                 OpAbstract::OpType::ASSIGN_OP); 

    for (Value v : funcOp.getArguments()) 
        ValueMap[v] = funcArgOp;

    // We need to manually add the constantOp which is not in the current
    // funcOp.
    auto designOp = llvm::dyn_cast<tor::DesignOp>(funcOp->getParentOp());
    for (auto &op : designOp.getBody()->getOperations())
      if (auto constOp = llvm::dyn_cast<ConstantOp>(op)) {
	OpAbstract *opA =
	  createOp(constOp, nullptr, bbs.first,
		   {constOp.getResult()},
		   SmallVector<Value>{});
	
	ValueMap[constOp.getResult()] = opA;
	bbs.first->addOperation(opA);
      }
    
    bbs.first->addOperation(funcArgOp);
    EntryBB = bbs.first, ExitBB = bbs.second;

    buildDFG();
  } 
}

void ScheduleBase::printCDFG() {
  for (unsigned i = 0, n = BasicBlocks.size(); i < n; ++i) {
    llvm::outs() << "BasicBlock " << i << ": " << BasicBlocks[i].get() << "\n";
    llvm::outs() << "===================================\n";

    for (auto op : BasicBlocks[i]->getOperations()) {
      if (op->getType() == OpAbstract::OpType::DEFINED_OP) {
        llvm::outs() << op->getOp()->getName().getStringRef().str() << ": \n";
      } else if (op->getType() == OpAbstract::OpType::PHI_OP) {
        llvm::outs() << "PHI: \n";
      } else if (op->getType() == OpAbstract::OpType::ASSIGN_OP) {
        llvm::outs() << "ASSIGN: \n";
      } else if (op->getType() == OpAbstract::OpType::LOAD_OP) {
	llvm::outs() << "LOAD: \n";
      } else if (op->getType() == OpAbstract::OpType::STORE_OP) {
	llvm::outs() << "STORE: \n";        
      }

      llvm::outs() << "operands(";
      for (auto opr : op->getOperands())
        llvm::outs() << mlir::hash_value(opr) << ", ";
      llvm::outs() << ")\n";

      llvm::outs() << "results(";
      for (auto res : op->getResults())
        llvm::outs() << mlir::hash_value(res) << ", ";
      llvm::outs() << ")\n";
    
      llvm::outs() << "ParentLoop: " << op->getParentLoop() << "\n";
    }

    llvm::outs() << "Successor BB: \n -------------------------\n";
    for (auto &succ : BasicBlocks[i]->getSucc()) {
      if (succ.type == ControlEdge::FORWARD)
        llvm::outs() << "Forward Edge: ";
      else if (succ.type == ControlEdge::LOOPBACK)
        llvm::outs() << "Loop Back Edge: ";
      else if (succ.type == ControlEdge::COND)
        llvm::outs() << "Condition Edge: ";
      llvm::outs() << succ.toBB << "\n";
    }
    llvm::outs() << "----------------------\n";

    llvm::outs() << "===================================\n";
  }

  llvm::outs() << "\n";
  llvm::outs() << "Data Dependencies: \n ===========================\n";
  for (auto &D : Dependencies) {
    llvm::outs() << D->SourceOp->getOp()->getName().getStringRef() << " -> " <<
        D->DestinationOp->getOp()->getName().getStringRef() <<
        "  Distance: " << D->Distance << "\n";
  }
  llvm::outs() << "=============================\n";
}

void ScheduleBase::printSchedule() {
  for (unsigned i = 0, n = BasicBlocks.size(); i < n; ++i) {
    llvm::outs() << "BasicBlock " << i << ": " << BasicBlocks[i].get() << "\n";
    llvm::outs() << "=====================================\n";

    for (auto op : BasicBlocks[i]->getOperations()) {
      if (op->getType() == OpAbstract::OpType::DEFINED_OP) {
        llvm::outs() << op->getOp()->getName().getStringRef().str() << ": operands(";
        for (auto opr : op->getOperands())
          llvm::outs() << mlir::hash_value(opr) << ", ";
        llvm::outs() << ")";
      } else if (op->getType() == OpAbstract::OpType::PHI_OP) {
        llvm::outs() << "PHI: operands(";
        for (auto opr : op->getOperands())
          llvm::outs() << mlir::hash_value(opr) << ", ";
        llvm::outs() << ")";
      } else if (op->getType() == OpAbstract::OpType::ASSIGN_OP) {
        llvm::outs() << "ASSIGN: operands(";
        for (auto opr : op->getOperands())
          llvm::outs() << mlir::hash_value(opr) << ", ";
        llvm::outs() << ")";
      }
      
      llvm::outs() << " at cycle " << op->getStartTime() << "\n";
    }

    llvm::outs() << "=====================================\n";
  }
}

LogicalResult ScheduleBase::verify() {
  return success();
}

} // namespace scheduling
