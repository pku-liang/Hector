#ifndef SCHEDULE_RESOURCEDB_H
#define SCHEDULE_RESOURCEDB_H

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"

#include "nlohmann/json.hpp"
#include <map>
#include <string>
#include <vector>

const int BIT_WIDTH_TYPE = 7;

namespace scheduling {

/// @brief A hardware component. Can either be a builtin component
///        or retrieved from a function without side effect.
struct Component {
  std::string name;         /// Name of the component.
  std::vector<float> delay; /// combinatorial delay of the component. Different
                            /// bitwidth may have different delay
  int latency;              /// latency of the component
  int II; /// Initial interval of a pipelined component. Equals to latency when
          /// not a pipeline function
  bool constr; /// should this components has constrained usage
  int amount;  /// Amount of resources of this kind. -1 when the resource can't
               /// be shared

  Component(const std::string &name, const std::vector<float> &delay,
            int latency, int II, bool constr = -1, int amount = -1)
      : name(name), delay(delay), latency(latency), II(II), constr(constr),
        amount(amount) {}

  Component(const std::string &name, float d, int latency, int II, bool constr,
            int amount)
      : name(name), latency(latency), II(II), constr(constr), amount(amount) {
    delay.resize(BIT_WIDTH_TYPE, d);
  }
};

/// @brief This class contains the allocation information for the scheduler
class ResourceDB {
public:
  int getResourceID(mlir::Operation *op) {
    auto name = op->getName().stripDialect().str();

    if (NameToID.find(name) != NameToID.end())
      return NameToID[name];

    if (auto callOp = llvm::dyn_cast<mlir::CallOp>(op)) {
      auto func = callOp.callee().str();
      if (NameToID.find(func) != NameToID.end())
        return NameToID[func];
    }

    return NameToID["nop"];
    // llvm::errs() << name << "\n";
    // assert(0 && "Error: Can't find corresponding resource");
  }

  int getResourceID(const std::string &str) {
    if (NameToID.find(str) != NameToID.end())
      return NameToID[str];

    assert(0 && "Error: Can't find corresponding resource");
  }

  int getII(int id) { return Components[id].II; }

  float getDelay(int id, int bitwidth) {
    int index = 0;
    switch (bitwidth) {
    case 64:
      index = 6;
      break;
    case 32:
      index = 5;
      break;
    case 16:
      index = 4;
      break;
    case 8:
      index = 3;
      break;
    case 4:
      index = 2;
      break;
    case 2:
      index = 1;
      break;
    default:
      index = 0;
    }

    return Components[id].delay[index];
  }

  std::string getName(int id) { return Components[id].name; }

  int getLatency(int id) { return Components[id].latency; }

  int getNumResource() { return Components.size(); }

  int getAmount(int id) { return Components[id].amount; }

  bool isCombLogic(int id) { return Components[id].latency == 0; }

  bool hasResConstr(int id) { return Components[id].constr == true; }

  bool hasHardLimit(int id) { return Components[id].amount != -1; }
  
  void addComponent(const Component &c) { Components.push_back(c); }

  ResourceDB() {
    NameToID.clear();
    Components.clear();
  }

  ResourceDB(nlohmann::json &config) {
    for (auto &res : config.items()) {
      auto info = res.value();

      std::string name = res.key();
      std::vector<float> delay;
      int latency = 0;
      int amount = -1;
      int II = 0;
      bool constr = 0;
      
      for (auto &item : info.items()) {
        if (item.key() == "delay") {
          for (auto &f : item.value().items())
            delay.push_back(f.value().get<float>());
        } else if (item.key() == "latency") {
          latency = item.value().get<int>();
        } else if (item.key() == "amount") {
          amount = item.value().get<int>();
        } else if (item.key() == "II") {
          II = item.value().get<int>();
        } else if (item.key() == "constr") {
	  constr = item.value().get<int>();
	}
      }

      addComponent(Component(name, delay, latency, II, constr, amount));
      NameToID[name] = Components.size() - 1;
    }
  }

private:
  std::map<std::string, int> NameToID;
  std::vector<Component> Components;
};

struct ScheduleTime {
public:
  int cycle, time;
  ScheduleTime(int c = 0, int t = 0) : cycle(c), time(t) {}

  ScheduleTime(const std::pair<int, int> &x) {
    cycle = x.first, time = x.second;
  }

  bool operator==(const ScheduleTime &x) const {
    return cycle == x.cycle && time == x.time;
  }
  bool operator<(const ScheduleTime &x) const {
    return cycle < x.cycle || (cycle == x.cycle && time < x.time);
  }
  bool operator>(const ScheduleTime &x) const { return x < *this; }
  bool operator<=(const ScheduleTime &x) const {
    return *this < x || *this == x;
  }
  bool operator>=(const ScheduleTime &x) const {
    return *this > x || *this == x;
  }
};

} // namespace scheduling

#endif
