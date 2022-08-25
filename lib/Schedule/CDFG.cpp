#include "Schedule/CDFG.h"

namespace scheduling {

bool canReach(OpAbstract *st, OpAbstract *en, bool backFlag) {
  std::unordered_map<BasicBlock *, bool> visited;
  std::queue<BasicBlock *> Q;

  if (st->getOp() == nullptr || en->getOp() == nullptr) {
    // one of these operation are created
    return false;
  }

  if (st->getParentBB() == en->getParentBB() && 
      st->getOp()->isBeforeInBlock(en->getOp()))
    return true;

  auto startBB = st->getParentBB();
  auto endBB = en->getParentBB();

  Q.push(startBB);

  while (!Q.empty()) {
    auto now = Q.front();
    Q.pop();

    if (visited[now] && now == endBB)
      return true;

    for (auto Succ : now->getSucc()) {
      if (backFlag == false && Succ.type == ControlEdge::LOOPBACK)
        continue;

      if (!visited[Succ.toBB]) {
        Q.push(Succ.toBB);
        visited[Succ.toBB] = true;
      }
    }
  }

  return false;
}

/// Chech whether two memory operation in DAG might have memory conflict
bool hasMemPortConflict(OpAbstract *PrevMemOp, OpAbstract *LatMemOp) {
  MemOpConcrete *memop_prev = PrevMemOp->getMemOp(),
                *memop_lat = LatMemOp->getMemOp();

  // assume different tensor uses different memory bank
  if (memop_prev->getMemRef() != memop_lat->getMemRef())
    return false;

  // TODO determine whether the two indices can be identical
  if (!memop_prev->hasFixedMemoryBank() || !memop_lat->hasFixedMemoryBank())
    return true;

  if (memop_prev->getMemoryBankIdx() == memop_lat->getMemoryBankIdx())
    return true;

  return false;
}

/// Check whehter two memory operation in the inner-most loop can have
/// memory port conflict with a distance Dist.
bool hasMemPortConflict(OpAbstract *PrevMemOp, OpAbstract *LatMemOp, int Dist) {
  // only inner-most loop for now
  if (PrevMemOp->getParentLoop() != LatMemOp->getParentLoop())
    return false;

  // const Loop *L = PrevMemOp->getParentLoop();

  MemOpConcrete *memop_prev = PrevMemOp->getMemOp(),
                *memop_lat = LatMemOp->getMemOp();

  // assume different tensor uses different memory bank
  if (memop_prev->getMemRef() != memop_lat->getMemRef())
    return false;

  // a very weak detection
  // TODO Add SCEV
  if (!memop_prev->hasFixedMemoryBank() || !memop_lat->hasFixedMemoryBank())
    return true;

  if (memop_prev->getMemoryBankIdx() == memop_lat->getMemoryBankIdx())
    return true;

  return false;
}

} // namespace scheduling
