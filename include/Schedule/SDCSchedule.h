#ifndef SCHEDULE_SDCSCHEDULE_H
#define SCHEDULE_SDCSCHEDULE_H

#include "ScheduleAlgo.h"
#include "SDCSolver.h"
#include "lp_lib.h"

namespace scheduling {

class SDCOpWrapper : public OpWrapperBase {
public:
  static bool classof(const OpAbstract *op) {
    return op->getKind() == OK_SDCWRAPPER;
  }

  SDCOpWrapper(OpAbstract *op) : 
  OpWrapperBase(op, OK_SDCWRAPPER)
  {}

  int getStartTime() override {
    return SDCTime;
  }

  int VarId, SDCTime, ASAPTime, OptTime;
};

class SDCSchedule : public ScheduleBase {
private:
  int resourceMII(Loop *L);

  int recurrenceMII(Loop *L);

  bool pipelineLoop(Loop *L);

  /**
   * @brief try to pipeline loop L with II
   * @param L loop to be pipelined
   * @param II II to achieve
   * @param FinalFlag Whether optimize register lifetime if schedule with II can be achieved.
   */
  bool scheduleWithII(Loop *L, int II, bool FinalFlag);
  
  void formulateDependency(Loop *L, int II, SDCSolver *SDC);

  void allocVariable(Loop *L, SDCSolver *SDC);

  /**
   * Brute-force algorithm to enumerate all possible path starting from op
   * Add constraint to SDC if exists one path exceed clock period constraint
   * @param op current op
   * @param SDC SDC system
   * @param cp current path delay
   * @param dist current path inter-iteration distance
   * @param II II to achieve
   * @param start starting op
   * @param vis Keep ops in current path to ensure simple path
   * @param exceed Use for pruning
   */
  void 
  traverse(SDCOpWrapper *op, SDCSolver *SDC, int latency, float cp, int dist, int II, 
      SDCOpWrapper* start, 
      std::unordered_map<SDCOpWrapper*, bool> &vis,
      std::unordered_map<SDCOpWrapper*, bool> &exceed);

  /**
   * Find critical paths starting from op in a loop.
   * Recurrence are considered. However, since cycles exist, finding
   * longest path between two node is NP Hard.
   * Currently, implement a exponential algorithm which enumerate all possible simple paths.
   */
  void addChainingConstr(SDCOpWrapper *op, SDCSolver *SDC, int II);

  bool optimizeASAP(Loop *L, int II, SDCSolver *SDC);
  
  /**
   * Get a reference solution without resource constraint as Described in the paper
   * This solution minimize the register pressure. Due to extra constraints added,
   * SDC solver can not solve this, a lp solver is needed.
   */
  bool minimizeLifetime(Loop *L, int II, SDCSolver *SDC);

  bool resolveResConstraint(Loop *L, int II, SDCSolver *SDC);

  bool getASAPTime(Loop *L, int II, SDCSolver *SDC);

  void printSDCSchedule(Loop *L);

  /**
   * Find critical paths starting from op. Only dependencies with distance 0 are considered.
   * Using SPFA.
   * @param op starting op
   * @param SDC System to add constraints in
   */
  void addTimingConstr(SDCOpWrapper *op, SDCSolver *SDC);

  std::vector<std::vector<int>>
  addResourceConstrBB(BasicBlock *BB, std::vector<std::vector<int>>&& Vars, int RId, SDCSolver *SDC);

  void addResourceConstr(int RId, SDCSolver *SDC);

  /// get a feasible total order of operations inside the basic block
  std::vector<SDCOpWrapper*> getFeasibleOrder(BasicBlock *BB, function_ref<bool(SDCOpWrapper*)> Pred);

  /// add memory resource constraint
  void addMemConstr(SDCSolver *SDC);

  SDCSolver *formulateSDC();

  bool minimizeLifetimeFunction(int II, SDCSolver *SDC);
  bool resolveResourceConstraintFunction(int II, SDCSolver *SDC);
  bool pipelineFunctionWithII(int II, bool FinalFlag);
  bool pipelineFunction();
public:
  explicit SDCSchedule(Operation *op) : ScheduleBase(op) {
    SDCOperations.clear();
  }
  
  LogicalResult runSchedule() override;
  
private:
  std::vector<std::unique_ptr<SDCOpWrapper>> SDCOperations;
};

} // namespace scheduling

#endif
