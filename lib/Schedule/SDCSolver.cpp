#include "Schedule/SDCSolver.h"
#include <unordered_map>
#include <queue>

namespace scheduling {

SDCSolver::SDCSolver() {
  Edges.resize(1, std::multiset<Edge>());
  Solution.resize(1);
  ValidFlag = 0;
  NumVariable = 1;
}

void SDCSolver::addInitialConstraint(Constraint C) {
  assert(C.x < NumVariable);
  assert(C.y < NumVariable);

//  C.dump();
  C.dump();
  if (C.type == Constraint::Constr_EQ) {
    Edges[0].insert(Edge(C.x, C.c));
    Edges[C.x].insert(Edge(0, -C.c));
  } if (C.type == Constraint::Constr_CMP)
    Edges[C.y].insert(Edge(C.x, C.c));
}

int SDCSolver::addVariable() {
  Edges.push_back(std::multiset<Edge>());
  Solution.push_back(0);
  Edges[0].insert(Edge(NumVariable++, 0)); // x >= 0
  return NumVariable - 1;
}

void SDCSolver::assignSolution(int x, int sol) {
  assert(x < NumVariable);
  Solution[x] = sol;
}

bool SDCSolver::initSolution() {
  Solution.resize(NumVariable, 0);
  int NumIteration = 0;
  
  do {
    bool UpdateFlag = false;

    for (int i = 0; i < NumVariable; ++i) 
      for (auto &edge : Edges[i]) 
        if (Solution[edge.to] < Solution[i] + edge.length) {
          Solution[edge.to] = Solution[i] + edge.length;
          UpdateFlag = true;
        }

    if (UpdateFlag == false)
    /// Converged. Must happens no late than the NumVariable-th iteration
      break;

    NumIteration++;
  } while (NumIteration < NumVariable);

  if (NumIteration == NumVariable)
    return ValidFlag =  false;
  return ValidFlag = true;
}

int SDCSolver::tryAddConstr(Constraint C) {

  int x = C.x, y = C.y, c = C.c; // Edge y -(c)-> x

  std::unordered_map<int, int> dist_x;
  std::unordered_map<int, int> NewSolution;

  auto cmp = [&] (int a, int b) {
    return dist_x[a] != dist_x[b] ? dist_x[a] < dist_x[b] : a < b;
  };

  bool valid = true;

  /// Dijkstra Algorithm
  std::priority_queue<int, std::vector<int>, decltype(cmp)> Q(cmp);

  dist_x[x] = 0;
  Q.push(x);

  int cnt = 0;

  while (!Q.empty()) {
    int now = Q.top();
    Q.pop();  

    int NewSol = Solution[y] + c + (dist_x[now] + Solution[now] - Solution[x]);
    if (Solution[now] < NewSol) {
      /// Solution[now] is affected
      if (now == y) {
        /// no feasible solution because there is a positive cycle
        valid = false;
        break;
      } else {
        NewSolution[now] = NewSol;

        for (auto &edge : Edges[now]) 
          if (dist_x.find(edge.to) == dist_x.end()) {
            dist_x[edge.to] = dist_x[now] + edge.length - 
                              Solution[edge.to] + Solution[now]; /// ensure negative edge
            Q.push(edge.to);
          }
      }
    }
  }

  if (valid == false)
    return -1;

  bool flag = false;
  if (NewSolution[0] > 0) {
    flag = true;

    int tmp = NewSolution[0];
    for (auto &iter : NewSolution)
      iter.second -= tmp;
  }

  for (auto &iter : NewSolution)
    if (Solution[iter.first] != iter.second)
      cnt++;

  if (flag)
    cnt += NumVariable - NewSolution.size();

  return cnt; 
}

bool SDCSolver::addConstraint(Constraint C) {

  if (C.type == Constraint::Constr_EQ) {
    if (addConstraint(Constraint::CreateGE(C.x, C.c)))
      return addConstraint(Constraint::CreateLE(C.x, C.c));
    return false;
  }

  int x = C.x, y = C.y, c = C.c; // Edge y -(c)-> x

  std::unordered_map<int, int> dist_x;
  std::unordered_map<int, int> NewSolution;

  auto cmp =
    [&] (std::pair<int, int> a, std::pair<int, int> b) {
      return a.second != b.second ? a.second < b.second : a.first < b.first;
    };
  
  bool valid = true;

  // Dijkstra Algorithm
  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, decltype(cmp)> Q(cmp);

  dist_x[x] = 0;
  Q.push(std::make_pair(x, 0));
  
  while (!Q.empty()) {
    int now = Q.top().first;
    dist_x[now] = Q.top().second;
    Q.pop();  

    int NewSol = Solution[y] + c + (dist_x[now] + Solution[now] - Solution[x]);
    
    if (Solution[now] < NewSol) {
      // Solution[now] is affected
      if (now == y) {
        // no feasible solution because there is a positive cycle
        valid = false;
        break;
      } else {

        NewSolution[now] = NewSol;

        for (auto &edge : Edges[now]) 
          if (dist_x.find(edge.to) == dist_x.end()) {
            int nxt_dist = dist_x[now] + edge.length - 
	      Solution[edge.to] + Solution[now]; // ensure negative edge
	    
            Q.push(std::make_pair(edge.to, nxt_dist));
          }
      }
    }
  }

  if (valid == false)
    return false;

  Edges[y].insert(Edge(x, c));
  for (auto &sol : NewSolution)
    Solution[sol.first] = sol.second;

  if (Solution[0] > 0) {
    int tmp = Solution[0];
    for (int i = 0; i < NumVariable; ++i) 
      Solution[i] -= tmp;
  }

  return true;
}

void SDCSolver::deleteConstraint(Constraint C) {
  if (C.type == Constraint::Constr_EQ) {
    deleteConstraint(Constraint::CreateGE(C.x, C.c));
    deleteConstraint(Constraint::CreateLE(C.x, C.c));
    return;
  }

  Edges[C.y].erase(Edge(C.x, C.c));
}

int SDCSolver::verify() {

  if (ValidFlag)
    llvm::outs() << "Has feasible solution\n";
  else {
    llvm::outs() << "No feasible solution\n";
    return 0;
  }

  int flag = 1;
  for (int i = 0; i < NumVariable; ++i)
    for (auto &edge : Edges[i]) {
      if (Solution[edge.to] - Solution[i] < edge.length) {
        llvm::outs() << "Require v" << edge.to << " - v" << i <<
            " >= " << edge.length << ". ";
        llvm::outs() << "While v" << edge.to << ": " << Solution[edge.to] << ", v" << 
            i << ": " << Solution[i] << "\n";

        flag = -1;
      }
    }
  
  return flag;
}

void SDCSolver::convertLP(lprec *lp) {
  // lp solve automatically deserve the bounds
  set_add_rowmode(lp, TRUE);

  int colno[2];
  REAL row[2];

  for (int i = 0; i < NumVariable; ++i)
    for (auto &edge : Edges[i]) {
      // edge.to - i >= edge.length
      // starting at 1
      colno[0] = edge.to + 1, row[0] = 1;
      colno[1] = i + 1, row[1] = -1;
      add_constraintex(lp, 2, row, colno, GE, edge.length);
    }

  set_add_rowmode(lp, FALSE);
}

void SDCSolver::printSolution() {
  if (ValidFlag == false) {
    llvm::outs() << "No feasible solution\n";
    return;
  }

  llvm::outs() << "Feasible solution\n";
  for (int i = 0; i < NumVariable; ++i)
    llvm::outs() << i << ": " << Solution[i] << "\n";
}

bool SDCSolver::isValid() {
  return ValidFlag;
}

void SDCUnittest() {
  // SDC Solver unit test
  {
    SDCSolver *SDC = new SDCSolver();
    int x1 = SDC->addVariable();
    int x2 = SDC->addVariable();
    int x3 = SDC->addVariable();
    int x4 = SDC->addVariable();
    int x5 = SDC->addVariable();
    SDC->addInitialConstraint(Constraint::CreateLE(x1, x2, 3));
    SDC->addInitialConstraint(Constraint::CreateLE(x3, x2, -2));
    SDC->addInitialConstraint(Constraint::CreateLE(x1, x3, 3));
    SDC->addInitialConstraint(Constraint::CreateLE(x3, x1, -3));
    SDC->addInitialConstraint(Constraint::CreateLE(x4, x3, -1));
    SDC->addInitialConstraint(Constraint::CreateLE(x5, x4, 4));
    // one feasible solution:(4, 4, 1, 0, 4)
    assert(SDC->initSolution());
    SDC->printSolution();
    assert(SDC->verify());
    assert(SDC->addConstraint(Constraint::CreateGE(x5, 1)));
    assert(SDC->addConstraint(Constraint::CreateGE(x4, 1)));
    SDC->printSolution();
  }
}

} // namespace scheduling
