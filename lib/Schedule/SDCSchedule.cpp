#include "Schedule/SDCSchedule.h"
#include "Schedule/CDFG.h"
#include "Schedule/SDCSolver.h"
#include "lp_lib.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <map>
#include <queue>
#include <set>
#include <unordered_map>

// Fix bug in Windows
#undef min
#undef max

namespace scheduling {

/**
 * Not scheduled in the pipeline scheduling
 */
bool needSchedule(const BasicBlock *B) {
  // return BeginBB.find(B) != BeginBB.end();
  return B->getParentLoop() == nullptr ||
         B->getParentLoop()->PipelineFlag == false;
}

int SDCSchedule::resourceMII(Loop *L) {
  int resII = 1;
  int ResourceKind = RDB.getNumResource();

  std::vector<int> resPressure(ResourceKind, 0);

  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      int RId = op->getResource();
      int pressure = RDB.getII(RId);

      resPressure[RId] += pressure;
    }

  for (int i = 1; i < ResourceKind; ++i)
    if (RDB.hasHardLimit(i)) {
      llvm::outs() << RDB.getName(i) << " " << resPressure[i] << " " << RDB.getAmount(i) << "\n";
      resII = std::max(resII, resPressure[i] / (int)RDB.getAmount(i));
    }

  return resII;
}

int SDCSchedule::recurrenceMII(Loop *L) {
  /// Why need this in the paper?
  int recII = 1;
  return recII;
}

bool sameLoop(Dependence *D) {
  return D->SourceOp->getParentLoop() == D->DestinationOp->getParentLoop();
}

void SDCSchedule::traverse(SDCOpWrapper *op, SDCSolver *SDC, int latency,
                           float cp, int dist, int II, SDCOpWrapper *start,
                           std::unordered_map<SDCOpWrapper *, bool> &vis,
                           std::unordered_map<SDCOpWrapper *, bool> &exceed) {
  vis[op] = true;
  for (auto Succ : op->getSucc())
    if (sameLoop(Succ)) {
      auto succOp = llvm::dyn_cast<SDCOpWrapper>(Succ->DestinationOp);
      float nxt_cp =
          cp + RDB.getDelay(succOp->getResource(), succOp->getWidth());
      int nxt_dist = dist + Succ->Distance;

      if (nxt_cp > ClockPeriod) {
        SDC->addInitialConstraint(Constraint::CreateGE(
            succOp->VarId, start->VarId, latency + 1 - nxt_dist * II)
                                  /* must schedule in different cycle */
        );
        exceed[succOp] = true;
        continue;
      }

      if (RDB.isCombLogic(succOp->getResource()) && !vis[succOp] &&
          exceed.find(succOp) == exceed.end())
        traverse(succOp, SDC, latency, nxt_cp, nxt_dist, II, start, vis,
                 exceed);
    }
  vis[op] = false;
}

void SDCSchedule::addChainingConstr(SDCOpWrapper *op, SDCSolver *SDC, int II) {
  std::unordered_map<SDCOpWrapper *, bool> vis;
  std::unordered_map<SDCOpWrapper *, bool> exceed;

  if (RDB.getName(op->getResource()) != "nop")
    traverse(op, SDC, RDB.getLatency(op->getResource()), 0.0, 0, II, op, vis,
             exceed);
}

void SDCSchedule::formulateDependency(Loop *L, int II, SDCSolver *SDC) {
  // formulate data dependency
  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      auto destOp = llvm::dyn_cast<SDCOpWrapper>(op);
      assert(destOp);

      for (auto pred : op->getPred())
        if (sameLoop(pred)) {
          auto srcOp = llvm::dyn_cast<SDCOpWrapper>(pred->SourceOp);
	  int RId = srcOp->getResource();
          int Lat = RDB.getLatency(RId);

	  // This special case is because of codegen backend
	  if (srcOp->getType() == OpAbstract::OpType::PHI_OP)
	    Lat = 1;
          if (Lat == 0) {
            if (RDB.getLatency(destOp->getResource()) == 0)
              SDC->addInitialConstraint(Constraint::CreateGE(
                  destOp->VarId, srcOp->VarId, -II * pred->Distance));
            else
              SDC->addInitialConstraint(Constraint::CreateGE(
                  destOp->VarId, srcOp->VarId, 1 - II * pred->Distance));
          } else {
            SDC->addInitialConstraint(Constraint::CreateGE(
                destOp->VarId, srcOp->VarId, Lat - II * pred->Distance));
          }
          // srcOp->printName(llvm::outs());
          // destOp->printName(llvm::outs());
          // llvm::outs() << Lat << " " << pred->Distance << "\n";
        }
    }

  // formulate chaining requirements
  for (auto BB : L->getBody())
    for (auto op : BB->getOperations())
      addChainingConstr(llvm::dyn_cast<SDCOpWrapper>(op), SDC, II);
}

void SDCSchedule::allocVariable(Loop *L, SDCSolver *SDC) {
  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      sdcOp->VarId = SDC->addVariable();
    }
}

bool SDCSchedule::optimizeASAP(Loop *L, int II, SDCSolver *SDC) {
  int varCnt = SDC->getNumVariable();

  lprec *lp = make_lp(0, SDC->getNumVariable());
  set_verbose(lp, NEUTRAL);
  SDC->convertLP(lp);

  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      set_mat(lp, 0, sdcOp->VarId, 1.0);
    }
  set_minim(lp);
  int ret = solve(lp);
  if (ret == INFEASIBLE)
    return false;

  REAL *results = new REAL[varCnt];
  get_variables(lp, results);

  for (int i = 0, n = SDC->getNumVariable(); i < n; ++i) {
    int sol = std::round(results[i]);
    assert(sol >= 0);
    SDC->assignSolution(i, sol);
  }

  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      sdcOp->OptTime = SDC->Solution[sdcOp->VarId];
    }

  SDC->ValidFlag = true;

  assert(SDC->verify());

  delete_lp(lp);

  return true;
}

bool SDCSchedule::minimizeLifetime(Loop *L, int II, SDCSolver *SDC) {
  int varCnt = SDC->getNumVariable();

  lprec *lp = make_lp(0, SDC->getNumVariable());
  set_verbose(lp, NEUTRAL);
  SDC->convertLP(lp);

  // add lifetime variables
  for (auto BB : L->getBody()) {
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      // add lifetime variable for op
      int rowno[1] = {0};
      double col[1] = {
          (double)sdcOp->getWidth()}; // coefficient of objective function
      add_columnex(lp, 0, col, rowno);

      varCnt++;
      int newId = get_Ncolumns(lp);

      set_add_rowmode(lp, TRUE);
      for (auto succ : op->getSucc())
        if (sameLoop(succ) && succ->type == Dependence::D_RAW) {
          auto succOp = llvm::dyn_cast<SDCOpWrapper>(succ->DestinationOp);

          int colno[3] = {succOp->VarId + 1, sdcOp->VarId + 1, newId};
          double row[3] = {1, -1, -1};
          add_constraintex(lp, 3, row, colno, LE,
                           RDB.getLatency(sdcOp->getResource()));
        }
      set_add_rowmode(lp, FALSE);
    }
  }

  set_minim(lp);
  int ret = solve(lp);
  if (ret == INFEASIBLE)
    return false;

  REAL *results = new REAL[varCnt];
  get_variables(lp, results);

  for (int i = 0, n = SDC->getNumVariable(); i < n; ++i) {
    int sol = std::round(results[i]);
    assert(sol >= 0);
    SDC->assignSolution(i, sol);
  }

  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      sdcOp->OptTime = SDC->Solution[sdcOp->VarId];
    }

  SDC->ValidFlag = true;

  assert(SDC->verify());

  delete_lp(lp);

  return true;
}

bool SDCSchedule::resolveResConstraint(Loop *L, int II, SDCSolver *SDC) {
  int ResKind = RDB.getNumResource();

  // keep the resource usage
  std::vector<std::vector<int>> ResTable(ResKind, std::vector<int>(II, 0));
  std::vector<int> ResLimit(ResKind, 0);
  std::vector<bool> HardFlag(ResKind, 0);
  
  for (int i = 0; i < ResKind; ++i) 
    if (RDB.hasResConstr(i)) {
      if (RDB.hasHardLimit(i)) {
	ResLimit[i] = RDB.getAmount(i);
	HardFlag[i] = 1;
      } else {
	int NumOp = 0;
	for (auto BB : L->getBody())
	  for (auto op : BB->getOperations())
	    if (op->getResource() == i)
	      NumOp++;
	ResLimit[i] = (NumOp + II - 1) / II;
      }
    }
  
  // keep the scheduled memop
  std::vector<std::vector<std::pair<SDCOpWrapper *, int>>> ScheduledMemOp(II);

  std::set<std::pair<int, SDCOpWrapper *>> S;

  // Ensure each op is tried at most II times.
  std::map<SDCOpWrapper *, int> failedCnt;

  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      if (RDB.hasResConstr(sdcOp->getResource()))
        S.insert(std::make_pair(sdcOp->OptTime, sdcOp));
    }

  while (!S.empty()) {
    int step = S.begin()->first;

    std::vector<SDCOpWrapper *> ops;
    while (!S.empty() && S.begin()->first == step) {
      ops.push_back(S.begin()->second);
      S.erase(S.begin());
    }

    // calc pertubation
    std::map<SDCOpWrapper *, int> pertubation;
    std::map<SDCOpWrapper *, int> earliestTime;

    for (auto op : ops) {
      int pert_before =
          SDC->tryAddConstr(Constraint::CreateLE(op->VarId, step - 1));
      int pert_after =
          SDC->tryAddConstr(Constraint::CreateGE(op->VarId, step + 1));
      pert_before = pert_before != -1 ? pert_before : SDC->getNumVariable();
      pert_after = pert_after != -1 ? pert_after : SDC->getNumVariable();

      pertubation[op] = std::max(pert_before, pert_after);
      earliestTime[op] = op->ASAPTime;
    }

    auto cmp = [&](SDCOpWrapper *a, SDCOpWrapper *b) {
      if (pertubation[a] != pertubation[b])
        return pertubation[a] > pertubation[b];
      return a->ASAPTime < b->ASAPTime;
    };

    std::sort(ops.begin(), ops.end(), cmp);

    for (auto op : ops) {
      bool scheduled = false;
      int RId = op->getResource();

      // only schedule resource constrained operations

      for (int s = step, n = earliestTime[op]; s >= n; --s) {
        // check resource availability
        bool avail = true;

        if (RDB.getName(RId) == "memport") {
          // assume memory port has one cycle latency and can't be pipelined
          int slot = s % II;
          for (auto &sdcOp : ScheduledMemOp[slot]) {
            int dist = (s - sdcOp.second) / II;
            // check if op in current iteration can have resource conflict with
            // sdc op after dist iteraions.

            if (hasMemPortConflict(op, sdcOp.first, dist)) {
              avail = false;
              break;
            }
          }
        } else {
          for (int i = 0; i < RDB.getII(RId); ++i)
            if (ResTable[RId][(s + i) % II] >= ResLimit[RId]) {
              avail = false;
              break;
            }
        }

        if (avail == false)
          continue;

        // TODO: Resource sharing among mutual exclusive branches.
        if (SDC->addConstraint(Constraint::CreateEQ(op->VarId, s))) {
          scheduled = true;

          for (int i = 0; i < RDB.getII(RId); ++i)
            ResTable[RId][(s + i) % II]++;

          if (RDB.getName(RId) == "memport")
            ScheduledMemOp[s % II].push_back(std::make_pair(op, s));

          break;
        }
      }

      if (scheduled == true)
        continue;

      if (SDC->addConstraint(Constraint::CreateGE(op->VarId, step + 1)) ==
          false)
        return false;

      if (failedCnt[op] == II)
        return false;

      failedCnt[op]++;
      earliestTime[op] = SDC->Solution[op->VarId];
      S.insert(std::make_pair(earliestTime[op], op));
    }
  }

  return true;
}

bool SDCSchedule::getASAPTime(Loop *L, int II, SDCSolver *SDC) {
  if (SDC->initSolution() == false)
    return false;

  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      sdcOp->ASAPTime = SDC->Solution[sdcOp->VarId];
    }

  return true;
}

bool SDCSchedule::scheduleWithII(Loop *L, int II, bool FinalFlag) {
  llvm::outs() << "Schedule with II: " << II << "\n";
  SDCSolver *SDC = new SDCSolver();

  allocVariable(L, SDC);

  formulateDependency(L, II, SDC);

  if (getASAPTime(L, II, SDC) == false)
    return false;

  if (minimizeLifetime(L, II, SDC) == false)
    return false;

  /*
  if (optimizeASAP(L, II, SDC) == false)
    return false;
  */

  if (resolveResConstraint(L, II, SDC) == false)
    return false;

  llvm::outs() << "Succceed\n";
  if (FinalFlag) {
    minimizeLifetime(L, II, SDC);

    assert(SDC->verify() == 1);

    for (auto BB : L->getBody())
      for (auto op : BB->getOperations()) {
        auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
        sdcOp->SDCTime = SDC->Solution[sdcOp->VarId];
      }
  }

  return true;
}

bool SDCSchedule::pipelineLoop(Loop *L) {
  if (L->PipelineFlag == false)
    return false;

  int recMII = recurrenceMII(L);
  int resMII = resourceMII(L);
  llvm::outs() << "recMII: " << recMII << " resMII: " << resMII << "\n";

  if (L->TargetII >= std::max(recMII, resMII)) {
    /// first try target II
    if (scheduleWithII(L, L->TargetII, 1)) {
      llvm::outs() << "Successful scheduled with TargetII: " << L->TargetII
                   << "\n";
      L->AchievedII = L->TargetII;
      return true;
    }
  }

  /// binary search
  int l = std::max(recMII, resMII), r = 64, achieved = -1;
  while (l <= r) {
    int mid = (l + r) >> 1;

    if (scheduleWithII(L, mid, 0)) {
      achieved = mid;
      r = mid - 1;
    } else {
      l = mid + 1;
    }
  }

  if (achieved != -1) {
    llvm::outs() << "Achieved II: " << achieved << "\n";
    L->AchievedII = achieved;
    scheduleWithII(L, achieved, 1);
    return true;
  } else {
    llvm::outs() << "Failed to find a reasonable II\n";
    return false;
  }
}

void SDCSchedule::addTimingConstr(SDCOpWrapper *op, SDCSolver *SDC) {
  std::queue<SDCOpWrapper *> Q;
  std::unordered_map<SDCOpWrapper *, bool> inQueue;
  std::unordered_map<SDCOpWrapper *, float> cp;

  int latency = RDB.getLatency(op->getResource());

  Q.push(op);
  cp[op] = RDB.getDelay(op->getResource(), op->getWidth());
  inQueue[op] = true;

  while (!Q.empty()) {
    auto now = Q.front();
    Q.pop();
    inQueue[now] = false;

    int dist = cp[now];

    for (auto Succ : now->getSucc()) {
      if (Succ->Distance != 0)
        continue;

      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(Succ->DestinationOp);

      if (!RDB.isCombLogic(sdcOp->getResource()))
        // critical path breaks
        continue;

      if (!needSchedule(sdcOp->getParentBB()))
        continue;

      if (cp.find(sdcOp) != cp.end() && cp[sdcOp] > ClockPeriod)
        continue;

      float nxt_dist =
          dist + RDB.getDelay(sdcOp->getResource(), sdcOp->getWidth());

      if (cp.find(sdcOp) == cp.end() || cp[sdcOp] < nxt_dist) {
        cp[sdcOp] = nxt_dist;
        if (inQueue[sdcOp] == false) {
          Q.push(sdcOp);
          inQueue[sdcOp] = true;
        }
      }

      if (cp[sdcOp] > ClockPeriod) {
        SDC->addInitialConstraint(
            Constraint::CreateGE(sdcOp->VarId, op->VarId, 1 + latency));
        continue;
      }
    }
  }
}

std::vector<SDCOpWrapper *>
SDCSchedule::getFeasibleOrder(BasicBlock *BB,
                              function_ref<bool(SDCOpWrapper *)> Pred) {
  std::unordered_map<OpAbstract *, int> InDeg;
  std::unordered_map<OpAbstract *, int> ASAP;
  std::unordered_map<OpAbstract *, float> Ord;
  std::queue<OpAbstract *> Q;

  for (auto op : BB->getOperations()) {
    for (auto Succ : op->getSucc())
      if (Succ->Distance == 0 && Succ->DestinationOp->getParentBB() == BB)
        InDeg[Succ->DestinationOp]++;
  }

  for (auto op : BB->getOperations())
    if (InDeg[op] == 0) {
      Q.push(op);
      ASAP[op] = 0;
      Ord[op] = RDB.getDelay(op->getResource(), op->getWidth());
    }

  while (!Q.empty()) {
    auto now = Q.front();
    Q.pop();

    for (auto Succ : now->getSucc())
      if (Succ->Distance == 0 && Succ->DestinationOp->getParentBB() == BB) {
        auto succOp = Succ->DestinationOp;
        ASAP[succOp] = std::max(
            ASAP[succOp], ASAP[now] + (int)RDB.getLatency(now->getResource()));
        Ord[succOp] =
            std::max(Ord[succOp], Ord[now] + RDB.getDelay(now->getResource(),
                                                          now->getWidth()));
        InDeg[succOp]--;
        if (InDeg[succOp] == 0)
          Q.push(succOp);
      }
  }

  std::vector<SDCOpWrapper *> ops;
  for (auto op : BB->getOperations()) {
    auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
    if (Pred(sdcOp))
      ops.push_back(sdcOp);
  }

  auto cmp = [&](SDCOpWrapper *a, SDCOpWrapper *b) {
    if (ASAP[a] != ASAP[b])
      return ASAP[a] < ASAP[b];
    return Ord[a] < Ord[b];
  };

  sort(ops.begin(), ops.end(), cmp);

  return ops;
}

std::vector<std::vector<int>>
SDCSchedule::addResourceConstrBB(BasicBlock *BB,
                                 std::vector<std::vector<int>> &&Vars, int RId,
                                 SDCSolver *SDC) {
  int Amount = RDB.getAmount(RId);

  std::vector<SDCOpWrapper *> constrainedOp = getFeasibleOrder(
      BB, [&](SDCOpWrapper *op) { return op->getResource() == RId; });

  std::vector<std::vector<int>> vec(std::move(Vars));

  for (int i = 0, n = constrainedOp.size(); i < n; ++i) {
    int slot = i % Amount;
    int Var = constrainedOp[i]->VarId;
    for (auto x : vec[slot])
      SDC->addInitialConstraint(Constraint::CreateGE(Var, x, 1));
    vec[slot] = {Var};
  }

  return vec;
}

void SDCSchedule::addResourceConstr(int RId, SDCSolver *SDC) {
  std::unordered_map<BasicBlock *, bool> Visited;
  std::unordered_map<BasicBlock *, int> InDeg;
  std::unordered_map<BasicBlock *, std::vector<std::vector<int>>> Chains;
  std::queue<BasicBlock *> Q;

  for (auto &&BB : BasicBlocks) {
    if (!needSchedule(BB.get()))
      continue;
    for (auto Succ : BB->getSucc())
      if (Succ.type != ControlEdge::LOOPBACK && needSchedule(Succ.toBB))
        InDeg[Succ.toBB]++;
  }

  int Amount = RDB.getAmount(RId);

  for (auto &&BB : BasicBlocks) {
    if (!needSchedule(BB.get()))
      continue;
    if (InDeg[BB.get()] == 0) {
      Chains[BB.get()].resize(Amount);
      Visited[BB.get()] = true;
      Q.push(BB.get());
    }
  }

  while (!Q.empty()) {
    auto now = Q.front();
    Q.pop();

    auto now_chains =
        addResourceConstrBB(now, std::move(Chains[now]), RId, SDC);

    for (auto Succ : now->getSucc())
      if (Succ.type != ControlEdge::LOOPBACK && needSchedule(Succ.toBB)) {
        auto succBB = Succ.toBB;
        if (!Visited[succBB]) {
          Chains[succBB].resize(Amount);
          Visited[succBB] = true;
        }

        auto &vec = Chains[succBB];
        for (int i = 0; i < Amount; ++i)
          vec[i].insert(vec[i].end(), now_chains[i].begin(),
                        now_chains[i].end());

        InDeg[succBB]--;
        if (InDeg[succBB] == 0)
          Q.push(succBB);
      }
  }
}

void SDCSchedule::addMemConstr(SDCSolver *SDC) {
  // assume no overlapping execution of basic blocks
  for (auto &&BB : BasicBlocks) {
    if (!needSchedule(BB.get()))
      continue;

    // resource confliction of pipelined loop has been handled
    // memport conflict of non-pipelined loop
    std::vector<SDCOpWrapper *> constrainedOp = getFeasibleOrder(
        BB.get(), [&](SDCOpWrapper *op) { return op->getMemOp() != nullptr; });
    for (int i = 0, n = constrainedOp.size(); i < n; ++i) {
      for (int j = i + 1; j < n; ++j)
        if (hasMemPortConflict(constrainedOp[i], constrainedOp[j])) {
          constrainedOp[i]->getOp()->dump();
          constrainedOp[j]->getOp()->dump();
          SDC->addInitialConstraint(Constraint::CreateGE(
              constrainedOp[j]->VarId, constrainedOp[i]->VarId, 1));
        }
    }
  }
}

SDCSolver *SDCSchedule::formulateSDC() {
  SDCSolver *SDC = new SDCSolver();

  std::unordered_map<BasicBlock *, int> BeginBB;
  std::unordered_map<BasicBlock *, int> EndBB;
  std::unordered_map<const Loop *, int> BeginLoop;
  std::unordered_map<const Loop *, int> EndLoop;

  // Allocate Variables
  for (auto &&BB : BasicBlocks) {
    if (!needSchedule(BB.get()))
      continue;

    BeginBB[BB.get()] = SDC->addVariable(); // super source of BB
    EndBB[BB.get()] = SDC->addVariable();   // super sink of BB

    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);

      sdcOp->VarId = SDC->addVariable();

      // super source -> op
      SDC->addInitialConstraint(
          Constraint::CreateGE(sdcOp->VarId, BeginBB[BB.get()], 0));
      SDC->addInitialConstraint(Constraint::CreateGE(
          EndBB[BB.get()], sdcOp->VarId, RDB.getLatency(sdcOp->getResource())));
    }
  }

  for (auto &&L : Loops)
    if (L->PipelineFlag) {
      BeginLoop[L.get()] = SDC->addVariable();
      EndLoop[L.get()] = SDC->addVariable();

      SDC->addInitialConstraint(
          Constraint::CreateGE(EndLoop[L.get()], BeginLoop[L.get()],
                               L->AchievedII /*IS IT SUFFICIENT?*/));
    }

  // Control Dependency
  for (auto &&BB : BasicBlocks) {
    if (!needSchedule(BB.get()))
      continue;

    for (auto Succ : BB->getSucc()) {
      // We currenly don't support overlapping execution of Basicblocks.

      if (Succ.type != ControlEdge::LOOPBACK) {
        auto succBB = Succ.toBB;

        if (needSchedule(succBB))
          SDC->addInitialConstraint(
              Constraint::CreateGE(BeginBB[succBB], EndBB[BB.get()], 1));
      }
    }

    for (auto Succ : BB->getSucc()) {
      if (Succ.type == ControlEdge::LOOPBACK)
        continue;
      if (needSchedule(Succ.toBB))
        continue;

      const Loop *L = Succ.toBB->getParentLoop();
      SDC->addInitialConstraint(
          Constraint::CreateGE(BeginLoop[L], EndBB[BB.get()], 1)
          /* Pipelined Loop can't share state with non loop block */
      );
    }

    for (auto Pred : BB->getPred()) {
      if (Pred.type == ControlEdge::LOOPBACK)
        continue;
      if (needSchedule(Pred.fromBB))
        continue;

      const Loop *L = Pred.fromBB->getParentLoop();
      SDC->addInitialConstraint(
          Constraint::CreateGE(BeginBB[BB.get()], EndLoop[L], 1)
          /* Pipelined Loop can't share state with non loop block */
      );
    }
  }

  // Data Dependency
  for (auto &&BB : BasicBlocks) {
    if (BeginBB.find(BB.get()) == BeginBB.end())
      continue;

    for (auto op : BB->getOperations()) {
      auto predOp = llvm::dyn_cast<SDCOpWrapper>(op);
      for (auto Succ : predOp->getSucc()) {

        if (Succ->Distance != 0)
          continue;

        auto succOp = llvm::dyn_cast<SDCOpWrapper>(Succ->DestinationOp);

        if (!needSchedule(succOp->getParentBB()))
          continue;

        SDC->addInitialConstraint(
            Constraint::CreateGE(succOp->VarId, predOp->VarId,
                                 RDB.getLatency(predOp->getResource())));
      }
    }
  }

  // Timing constraints
  for (auto &&BB : BasicBlocks)
    if (needSchedule(BB.get())) {
      for (auto op : BB->getOperations()) {
        auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
        addTimingConstr(sdcOp, SDC);
      }
    }

  // Resource constraints;
  int NumResource = RDB.getNumResource();

  for (int i = 0; i < NumResource; ++i)
    if (RDB.hasHardLimit(i))
      addResourceConstr(i, SDC);

  addMemConstr(SDC);

  return SDC;
}

bool SDCSchedule::minimizeLifetimeFunction(int II, SDCSolver *SDC) {
  int varCnt = SDC->getNumVariable();

  lprec *lp = make_lp(0, SDC->getNumVariable());
  set_verbose(lp, NEUTRAL);
  SDC->convertLP(lp);

  // add lifetime variables
  for (auto &&sdcOp : SDCOperations) {
    // add lifetime variable for op
    int rowno[1] = {0};
    // coefficient of objective function
    double col[1] = {(double)sdcOp->getWidth()};
    add_columnex(lp, 1, col, rowno);

    varCnt++;
    int newId = get_Ncolumns(lp);

    set_add_rowmode(lp, TRUE);
    for (auto succ : sdcOp->getSucc())
      if (succ->type == Dependence::D_RAW) {
        auto succOp = llvm::dyn_cast<SDCOpWrapper>(succ->DestinationOp);

        int colno[3] = {succOp->VarId + 1, sdcOp->VarId + 1, newId};
        double row[3] = {1, -1, -1};
        add_constraintex(lp, 3, row, colno, LE,
                         RDB.getLatency(sdcOp->getResource()));
      }
    set_add_rowmode(lp, FALSE);
  }

  set_minim(lp);
  int ret = solve(lp);
  llvm::outs() << ret << "\n";
  if (ret == INFEASIBLE)
    return false;

  REAL *results = new REAL[varCnt];
  get_variables(lp, results);

  for (int i = 0, n = SDC->getNumVariable(); i < n; ++i) {
    int sol = std::round(results[i]);
    assert(sol >= 0);
    SDC->assignSolution(i, sol);
  }

  for (auto &&sdcOp : SDCOperations)
    sdcOp->OptTime = SDC->Solution[sdcOp->VarId];

  SDC->ValidFlag = true;

  assert(SDC->verify() > 0);

  delete_lp(lp);

  return true;
}

bool SDCSchedule::resolveResourceConstraintFunction(int II, SDCSolver *SDC) {
  int ResKind = RDB.getNumResource();

  // keep the resource usage
  std::vector<std::vector<int>> ResTable(ResKind, std::vector<int>(II, 0));
  std::vector<int> ResLimit(ResKind, 0);
  std::vector<bool> HardFlag(ResKind, 0);
  
  for (int i = 0; i < ResKind; ++i) 
    if (RDB.hasResConstr(i)) {
      if (RDB.hasHardLimit(i)) {
	ResLimit[i] = RDB.getAmount(i);
	HardFlag[i] = 1;
      } else {
	int NumOp = 0;
	for (auto &&op : SDCOperations)
	  if (op->getResource() == i)
	    NumOp++;
	ResLimit[i] = (NumOp + II - 1) / II;
      }
    }
  
  // keep the scheduled memop
  std::vector<std::vector<std::pair<SDCOpWrapper *, int>>> ScheduledMemOp(II);

  std::set<std::pair<int, SDCOpWrapper *>> S;

  // Ensure each op is tried at most II times.
  std::map<SDCOpWrapper *, int> failedCnt;

  for (auto &&sdcOp : SDCOperations)
    if (RDB.hasResConstr(sdcOp->getResource()))
      S.insert(std::make_pair(sdcOp->OptTime, sdcOp.get()));

  while (!S.empty()) {
    int step = S.begin()->first;

    std::vector<SDCOpWrapper *> ops;
    while (!S.empty() && S.begin()->first == step) {
      ops.push_back(S.begin()->second);
      S.erase(S.begin());
    }

    // calc pertubation
    std::map<SDCOpWrapper *, int> pertubation;
    std::map<SDCOpWrapper *, int> earliestTime;

    for (auto op : ops) {
      int pert_before =
          SDC->tryAddConstr(Constraint::CreateLE(op->VarId, step - 1));
      int pert_after =
          SDC->tryAddConstr(Constraint::CreateGE(op->VarId, step + 1));
      pert_before = pert_before != -1 ? pert_before : SDC->getNumVariable();
      pert_after = pert_after != -1 ? pert_after : SDC->getNumVariable();

      pertubation[op] = std::max(pert_before, pert_after);
      earliestTime[op] = op->ASAPTime;
    }

    auto cmp = [&](SDCOpWrapper *a, SDCOpWrapper *b) {
      if (pertubation[a] != pertubation[b])
        return pertubation[a] > pertubation[b];
      return a->ASAPTime < b->ASAPTime;
    };

    std::sort(ops.begin(), ops.end(), cmp);

    for (auto op : ops) {
      bool scheduled = false;
      int RId = op->getResource();

      // only schedule resource constrained operations

      for (int s = step, n = earliestTime[op]; s >= n; --s) {
        // check resource availability
        bool avail = true;

        if (RDB.getName(RId) == "memport") {
          // assume memory port has one cycle latency and can't be pipelined
          int slot = s % II;
          for (auto &sdcOp : ScheduledMemOp[slot]) {
            int dist = (s - sdcOp.second) / II;
            // check if op in current iteration can have resource conflict with
            // sdc op after dist iteraions.

            if (hasMemPortConflict(op, sdcOp.first, dist)) {
              avail = false;
              break;
            }
          }
        } else {
          for (int i = 0; i < RDB.getII(RId); ++i)
            if (ResTable[RId][(s + i) % II] >= ResLimit[RId]) {
              avail = false;
              break;
            }
        }

        if (avail == false)
          continue;

        // TODO: Resource sharing among mutual exclusive branches.
        if (SDC->addConstraint(Constraint::CreateEQ(op->VarId, s))) {
          scheduled = true;

          for (int i = 0; i < RDB.getII(RId); ++i)
            ResTable[RId][(s + i) % II]++;

          if (RDB.getName(RId) == "memport")
            ScheduledMemOp[s % II].push_back(std::make_pair(op, s));

          break;
        }
      }

      if (scheduled == true)
        continue;

      if (SDC->addConstraint(Constraint::CreateGE(op->VarId, step + 1)) ==
          false)
        return false;

      if (failedCnt[op] == II)
        return false;

      failedCnt[op]++;
      earliestTime[op] = SDC->Solution[op->VarId];
      S.insert(std::make_pair(earliestTime[op], op));
    }
  }

  return true;
}

bool SDCSchedule::pipelineFunctionWithII(int II, bool FinalFlag) {
  llvm::outs() << "try II=" << II << "\n";
  SDCSolver *SDC = new SDCSolver();
  std::unordered_map<BasicBlock *, int> BeginBB;
  std::unordered_map<BasicBlock *, int> EndBB;

  // contol dependency
  for (auto &&BB : BasicBlocks) {
    BeginBB[BB.get()] = SDC->addVariable();
    EndBB[BB.get()] = SDC->addVariable();

    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      sdcOp->VarId = SDC->addVariable();
      SDC->addInitialConstraint(
          Constraint::CreateGE(sdcOp->VarId, BeginBB[BB.get()], 0));
      SDC->addInitialConstraint(Constraint::CreateGE(
          EndBB[BB.get()], sdcOp->VarId, RDB.getLatency(sdcOp->getResource())));
    }
  }

  for (auto &&BB : BasicBlocks) {
    for (auto Succ : BB->getSucc()) {
      // can't have loop inside function
      assert(Succ.type != ControlEdge::LOOPBACK &&
             "Can't have loop in the pipelined function");
      auto succBB = Succ.toBB;
      SDC->addInitialConstraint(
          Constraint::CreateGE(BeginBB[succBB], EndBB[BB.get()], 1));
    }
  }

  for (auto &&op : SDCOperations) {
    auto predOp = op.get();
    for (auto Succ : predOp->getSucc()) {
      auto succOp = llvm::dyn_cast<SDCOpWrapper>(Succ->DestinationOp);

      int Lat = RDB.getLatency(predOp->getResource());
      // This special case is because of codegen backend
      if (predOp->getType() == OpAbstract::OpType::PHI_OP)
	Lat = 1;
      SDC->addInitialConstraint(Constraint::CreateGE(
          succOp->VarId, predOp->VarId, Lat));
    }
  }

  for (auto &&BB : BasicBlocks)
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      addTimingConstr(sdcOp, SDC);
    }

  // get asap time
  SDC->initSolution();
  for (auto &&op : SDCOperations)
    op->ASAPTime = SDC->Solution[op->VarId];
  
  if (minimizeLifetimeFunction(II, SDC) == false)
    return false;
  
  for (auto &&sdcOp : SDCOperations) {
    llvm::outs() << sdcOp->getOp()->getName() << " "
		 << sdcOp->ASAPTime << " "
		 << sdcOp->OptTime << "\n";
  }
      
  if (resolveResourceConstraintFunction(II, SDC) == false)
    return false;
  if (FinalFlag) {
    minimizeLifetimeFunction(II, SDC);
    for (auto &&op : SDCOperations) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op.get());
      sdcOp->SDCTime = SDC->Solution[sdcOp->VarId];
    }
  }
  return true;
}

bool SDCSchedule::pipelineFunction() {
  int targetII = containingOp->getAttrOfType<IntegerAttr>("II").getInt();
  containingOp->removeAttr("II");
  if (pipelineFunctionWithII(targetII, false)) {
    llvm::outs() << "Achieved II: " << targetII << "\n";
    containingOp->setAttr(
        "II", IntegerAttr::get(IntegerType::get(containingOp->getContext(), 32),
                               targetII));    
    pipelineFunctionWithII(targetII, true);
    return true;
  }

  int l = targetII, r = 64, achieved = -1;
  while (l <= r) {
    int mid = (l + r) >> 1;

    if (pipelineFunctionWithII(mid, 0)) {
      achieved = mid;
      r = mid - 1;
    } else {
      l = mid + 1;
    }
  }

  if (achieved != -1) {
    llvm::outs() << "Achieved II: " << achieved << "\n";
    containingOp->setAttr(
        "II", IntegerAttr::get(IntegerType::get(containingOp->getContext(), 32),
                               achieved));
    pipelineFunctionWithII(achieved, true);
    return true;
  } else {
    llvm::outs() << "Failed to find a reasonable II\n";
    return false;
  }

  return false;
}

LogicalResult SDCSchedule::runSchedule() {
  buildFromContaingOp();

  SDCOperations = initSchedule<SDCOpWrapper>();
  
  if (auto pipeline_flag = containingOp->getAttrOfType<StringAttr>("pipeline")) {
    // pipeline this function
    if (pipeline_flag.getValue().str() == "func") {
      if (pipelineFunction())
	return success();
      llvm::outs() << containingOp->getName() << ". Function pipelining failed!\n";
      return failure();
    }
  }

  for (auto &&L : Loops) {
    if (L->PipelineFlag == true) {
      if (pipelineLoop(L.get()) == false)
        L->PipelineFlag = false;
    }
  }

  SDCSolver *SDC = formulateSDC();
  assert(SDC->initSolution());
  assert(SDC->verify());

  for (auto &&sdcOp : SDCOperations)
    if (sdcOp->getParentLoop() == nullptr ||
        sdcOp->getParentLoop()->PipelineFlag == false)
      sdcOp->SDCTime = SDC->Solution[sdcOp->VarId];

  return success();
}

void SDCSchedule::printSDCSchedule(Loop *L) {
  llvm::outs() << "Loop: " << L << "\n";
  llvm::outs() << "========================================\n";

  llvm::outs() << "Target II: " << L->TargetII << "\n";
  llvm::outs() << "Achieved II: " << L->AchievedII << "\n";

  for (auto BB : L->getBody()) {
    llvm::outs() << "BasicBlock: " << BB << "\n";

    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);

      if (op->getType() == OpAbstract::OpType::DEFINED_OP) {
        llvm::outs() << op->getOp()->getName().getStringRef().str()
                     << ": operands(";
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

      llvm::outs() << " at cycle " << sdcOp->SDCTime << "\n";
    }
  }

  llvm::outs() << "===========================================\n";
}

} // namespace scheduling
