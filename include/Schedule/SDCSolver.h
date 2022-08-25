#ifndef SDC_SOLVER_H
#define SDC_SOLVER_H

#include "llvm/Support/raw_ostream.h"

#include "mlir/Support/LLVM.h"

#include "lp_lib.h"
#include <set>
#include <vector>

namespace scheduling {

class Constraint {
public:
  enum Type {
    Constr_EQ, /// x = c
    Constr_CMP /// x - y >= c
  } type;
  int x, y, c;
  static Constraint CreateEQ(int x, int c) {
    return {Constr_EQ, x, 0, c};
  }
  static Constraint CreateGE(int x, int c) {
    return {Constr_CMP, x, 0, c};
  }
  static Constraint CreateLE(int x, int c) {
    return {Constr_CMP, 0, x, -c};
  }
  static Constraint CreateGE(int x, int y, int c) {
    return {Constr_CMP, x, y, c};
  }
  static Constraint CreateLE(int x, int y, int c) {
    return {Constr_CMP, y, x, -c};
  }
  void dump() {
    if (type == Constr_CMP)
      llvm::outs() << "v" << x << " - " << "v" << y << " >= " << c << "\n";
    else if (type == Constr_EQ)
      llvm::outs() << "v" << x << " = " << c << "\n";
  }
};

/**
 * This is a Solver for System of Difference Constraints.
 * Constraints are in the form x - y >= c -- x, y are variables and c is a constant.
 * Additional constraints is that variables must be non-negative.
 * Using bellman-ford on longest path to compute initial solution.
 * Using incremental algorithm described in "Solving Systems of Difference Constraints Incrementally"
 *  to incrementally add constraints (which is required by sdc-based modulo scheduling).
 */
class SDCSolver {
public:
  struct Edge {
    int to, length;

    Edge(int to, int l) : to(to), length(l) {}

    bool operator < (const Edge &x) const {
      return length != x.length ? length > x.length : to < x.to;
    }
  };

  std::vector<std::multiset<Edge>> Edges;
  std::vector<int> Solution;
public:
  SDCSolver();
  /**
   * Add initial constraint into SDC. x - y >= c
   * @param x
   * @param y
   * @param c
   */
  void addInitialConstraint(Constraint C);

  /**
   * Tentatively add a constraint C into the system.
   * @param C constraint. Must not be a EQ constraint.
   * @return number of variables affected by this constraint. -1 if invalid
   */
  int tryAddConstr(Constraint C);

  /**
   * Incrementally add constraint into SDC. x - y >= c
   * If the no feasible solution after constraint was added, this constraint won't be added.
   * @param x
   * @param y
   * @param c
   * @return Whether the solution is feasible
   */
  bool addConstraint(Constraint C);

  /**
   * Delete a constraint x - y >= c. Simple erase the edge, since solution remains valid.
   * @param x
   * @param y
   * @param c
   */
  void deleteConstraint(Constraint C);

  /**
   * Manually assign a solution. User must ensure the validity
   */
  void assignSolution(int x, int sol);

  /**
   * Allocate new variable
   * @return new variable id
   */
  int addVariable();

  int getNumVariable() { return NumVariable; }

  int getNumConstraint() { return NumConstraint; }

  // convert the constraints into lp
  void convertLP(lprec *lp);

  /**
   * Solve the initial constraints
   * @return Whether exists a feasible solution
   */
  bool initSolution();

  bool isValid();
  
  void printSolution();

  int verify();
private:

  int NumVariable; /// Number of Variables

  int NumConstraint; /// Number of Constraints

  bool ValidFlag; /// Does current SDC have feasible solution?
  friend class SDCSchedule;
};

} // namespace scheduling

#endif
