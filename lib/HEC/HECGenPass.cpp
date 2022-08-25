#include <mlir/Transforms/DialectConversion.h>

#include <algorithm>
#include <bitset>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <vector>

#include "TOR/TOR.h"
#include "TOR/TORDialect.h"
#include "HEC/HEC.h"
#include "HEC/HECDialect.h"
#include "HEC/PassDetail.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "generate-hec"

namespace mlir {
namespace hecgen {
struct TimeNode;
struct OpOnEdge;
struct TimeEdge;

class Component;

struct TimeNode {
  uint64_t t, tend, tthen, telse;

  enum class NodeT { SEQ, CALL, IF, FOR, ENDFOR, WHILE, ENDWHILE } type;

  mlir::Operation *op;
  mlir::Value val, val0, val1, val2, val3;
  std::vector<TimeEdge> edges;
  std::vector<TimeEdge> backEdges;
  std::vector<OpOnEdge> opsOnEdge;
  TimeNode(uint64_t t, NodeT type = NodeT::SEQ) : t(t), tend(t), type(type) {
    op = nullptr;
    val = nullptr;
    val0 = nullptr;
    val1 = nullptr;
    val2 = nullptr;
    val3 = nullptr;
    edges.clear();
    backEdges.clear();
    opsOnEdge.clear();
  }

  int64_t getPipelineII() {
    auto forop = llvm::dyn_cast<tor::ForOp>(op);
    assert(forop != nullptr);
    auto pipelineAttr = forop->getAttrOfType<IntegerAttr>("pipeline");
    if (pipelineAttr == nullptr)
      return -1;
    auto pipeline = pipelineAttr.getInt();
    if (!pipeline)
      return -1;

    auto IIAttr = forop->getAttrOfType<IntegerAttr>("II");
    assert(IIAttr != nullptr);
    return IIAttr.getInt();
  }
};

struct TimeEdge {
  uint64_t from;
  uint64_t to;
  mlir::Attribute attr;
  bool loop_exit;
  enum class SD { STATIC, DYNAMIC } sd;
  bool valid;
  TimeEdge(int f, int t, mlir::Attribute at, SD s, bool le = false)
      : from(f), to(t), attr(at), loop_exit(le), sd(s), valid(true) {}
};

struct OpOnEdge {
  uint64_t from;
  uint64_t to;
  mlir::Operation *op;

  OpOnEdge(uint64_t from, uint64_t to, mlir::Operation *op)
      : from(from), to(to), op(op) {}
};

struct ValueUseDef {
  size_t id;
  mlir::Value value;
  mlir::Value wire;
  mlir::Operation *owner;
  size_t def;              // def Node id
  std::string state;       // produce state
  std::vector<size_t> use; // vector of use Node id

  enum class Type {
    Constant,
    Memory,
    Variable,
    GlobalConstant,
    GlobalMem,
    GlobalVar,
    Argument,
    NoVal
  } type;

  ValueUseDef() = default;
  ValueUseDef(size_t id, mlir::Value value, mlir::Operation *owner, size_t def,
              Type type = Type::Variable)
      : id(id), value(value), wire(nullptr), owner(owner), def(def), state(""),
        type(type) {
    use.clear();
  }
};

struct Memory {
  size_t id;
  enum Type { R, RW, W } type;
  hec::PrimitiveOp op;
  Memory(size_t id, Type type = Type::RW, hec::PrimitiveOp op = nullptr)
      : id(id), type(type), op(op) {}

  mlir::Value getAddress() {
    assert(op != nullptr && op.primitiveName() == "mem");
    auto rw = op->getAttr("ports").cast<mlir::StringAttr>().getValue();
    if (rw == "r") {
      return op.getResult(1);
    } else if (rw == "w") {
      return op.getResult(1);
    } else if (rw == "rw") {
      return op.getResult(2);
    } else
      assert(0 && "No address");
    return nullptr;
  }
  mlir::Value getReadEnable() {
    assert(op != nullptr && op.primitiveName() == "mem");
    auto rw = op->getAttr("ports").cast<mlir::StringAttr>().getValue();
    if (rw == "r") {
      return op.getResult(0);
    } else if (rw == "rw") {
      return op.getResult(1);
    } else
      assert(0 && "No Read Enable");
    return nullptr;
  }
  mlir::Value getWriteEnable() {
    assert(op != nullptr && op.primitiveName() == "mem");
    auto rw = op->getAttr("ports").cast<mlir::StringAttr>().getValue();
    if (rw == "w" || rw == "rw") {
      return op.getResult(0);
    } else
      assert(0 && "No Write Enable");
    return nullptr;
  }
  mlir::Value getReadData() {
    assert(op != nullptr && op.primitiveName() == "mem");
    auto rw = op->getAttr("ports").cast<mlir::StringAttr>().getValue();
    if (rw == "r") {
      return op.getResult(2);
    } else if (rw == "rw") {
      return op.getResult(4);
    } else
      assert(0 && "No Read Data");
    return nullptr;
  }
  mlir::Value getWriteData() {
    assert(op != nullptr && op.primitiveName() == "mem");
    auto rw = op->getAttr("ports").cast<mlir::StringAttr>().getValue();
    if (rw == "w") {
      return op.getResult(2);
    } else if (rw == "rw") {
      return op.getResult(3);
    } else
      assert(0 && "No Write Data");
    return nullptr;
  }
};

struct Register {
  size_t id;
  mlir::Type type;
  hec::PrimitiveOp op;

  Register(size_t id, mlir::Type type = nullptr, hec::PrimitiveOp op = nullptr)
      : id(id), type(type), op(op) {}
};

struct Cell {
  size_t id;
  size_t t, tend;
  std::string cname;
  int suffix;
  std::string pname;
  hec::PrimitiveOp primitive;
  llvm::SmallVector<mlir::Type, 4> types;
  Cell(size_t id, size_t t, size_t tend, std::string cname, std::string pname,
       hec::PrimitiveOp op = nullptr)
      : id(id), t(t), tend(tend), cname(cname), suffix(-1), pname(pname),
        primitive(op) {
    types.clear();
  }
  template <typename OpType> void setTypes(OpType op) {

    for (size_t i = 0; i < op->getNumOperands(); i++)
      types.push_back(op->getOperand(i).getType());
    types.push_back(op.getResult().getType());
  }
};

struct GlobalStorage {
  std::vector<ValueUseDef> valueUseDefs;
  size_t value_count = 0;

  std::vector<Memory> memories;
  std::vector<bool> memused;
  size_t mem_count = 0;
  std::map<size_t, size_t> value2Mem;
  mlir::Operation *last = nullptr;

  std::vector<Register> registers;
  size_t reg_count = 0;
  std::map<size_t, size_t> value2Reg;

  mlir::hec::DesignOp design;
  mlir::PatternRewriter &rewriter;

  GlobalStorage(mlir::hec::DesignOp design, mlir::PatternRewriter &rewriter)
      : design(design), rewriter(rewriter) {}

  void add_constant(mlir::ConstantOp constantop) {
    ValueUseDef vud(value_count++, constantop.getResult(), constantop, -3,
                    ValueUseDef::Type::GlobalConstant);
    if (last == nullptr)
      rewriter.setInsertionPointToStart(design.getBody());
    else
      rewriter.setInsertionPoint(last);
    auto new_constant_op = rewriter.create<mlir::ConstantOp>(
        design.getLoc(), constantop.getValue());
    vud.owner = new_constant_op;
    last = new_constant_op;
    valueUseDefs.push_back(vud);
  }

  void add_reg(ValueUseDef vud) {
    vud.def = -3;
    vud.id = value_count++;
    valueUseDefs.push_back(vud);
    Register reg(reg_count++);

    llvm::SmallVector<mlir::Type, 1> types;
    types.push_back(vud.value.getType());

    auto cname = rewriter.getStringAttr(std::string("r_global_") +
                                        std::to_string(reg.id));
    auto pname = rewriter.getStringAttr("register");
    if (last == nullptr)
      rewriter.setInsertionPointToStart(design.getBody());
    else
      rewriter.setInsertionPointAfter(last);
    auto primitive = rewriter.create<hec::PrimitiveOp>(
        design.getLoc(), mlir::TypeRange(types), cname, pname);
    reg.op = primitive;
    registers.push_back(reg);
    value2Reg[vud.id] = reg.id;
    last = primitive;
  }

  void add_mem(tor::AllocOp alloc) {
    auto value = alloc.getResult();
    ValueUseDef vud(value_count++, value, alloc, -3,
                    ValueUseDef::Type::GlobalMem);
    valueUseDefs.push_back(vud);

    Memory mem(mem_count++);
    auto mem_type = alloc.getType();
    auto rw = mem_type.getRw();
    auto shape = mem_type.getShape();
    assert(shape.size() == 1 && "Only support 1-dim memory");
    auto length = shape.front();
    auto addressLen = static_cast<unsigned long>(ceil(log2(length)));
    auto elementType = mem_type.getElementType();
    llvm::SmallVector<mlir::Type> portTypes;

    if (rw.getValue() == "r") {
      portTypes.push_back(rewriter.getI1Type());
      portTypes.push_back(rewriter.getIntegerType(addressLen));
      portTypes.push_back(elementType);
      mem.type = Memory::Type::R;
    } else if (rw.getValue() == "w") {
      portTypes.push_back(rewriter.getI1Type());
      portTypes.push_back(rewriter.getIntegerType(addressLen));
      portTypes.push_back(elementType);
      mem.type = Memory::Type::W;
    } else if (rw.getValue() == "rw") {
      portTypes.push_back(rewriter.getI1Type());
      portTypes.push_back(rewriter.getI1Type());
      portTypes.push_back(rewriter.getIntegerType(addressLen));
      portTypes.push_back(elementType);
      portTypes.push_back(elementType);
      mem.type = Memory::Type::RW;
    }

    auto cname = rewriter.getStringAttr(std::string("mem_global_") +
                                        std::to_string(mem_count - 1));
    auto pname = rewriter.getStringAttr(std::string("mem"));

    if (last == nullptr)
      rewriter.setInsertionPointToStart(design.getBody());
    else
      rewriter.setInsertionPointAfter(last);

    auto primitive = rewriter.create<hec::PrimitiveOp>(
        design.getLoc(),
        llvm::ArrayRef<mlir::Type>(portTypes.begin(), portTypes.end()), cname,
        pname);
    primitive->setAttr("len", rewriter.getI32IntegerAttr(length));
    primitive->setAttr("ports", rw);

    mem.op = primitive;
    memories.push_back(mem);
    memused.push_back(0);
    value2Mem[vud.id] = mem.id;
    last = primitive;
  }

  mlir::Value getConstant(mlir::Value value) {
    for (auto vud : valueUseDefs)
      if (vud.value == value) {
        if (vud.type == ValueUseDef::Type::GlobalConstant)
          return llvm::cast<ConstantOp>(vud.owner).getResult();
      }
    return nullptr;
  }

  mlir::Value getReg(mlir::Value value) {
    for (auto vud : valueUseDefs)
      if (vud.value == value) {
        if (vud.type == ValueUseDef::Type::GlobalVar)
          return registers[value2Reg[vud.id]].op.getResults().front();
        // else
        //   assert(0 && "Unmatched Type for Global ValueUseDef");
      }
    // assert(0 && "Global Reg not found");
    return nullptr;
  }

  Memory getMem(mlir::Value value) {
    for (auto vud : valueUseDefs)
      if (vud.value == value) {
        if (vud.type == ValueUseDef::Type::GlobalMem) {
          memused[value2Mem[vud.id]] = 1;
          return memories[value2Mem[vud.id]];
        }
        // else
        //   assert(0 && "Unmatched Type for Global ValueUseDef");
      }
    // assert(0 && "Global Mem not found");
    return Memory(-1);
  }
  void codegen(mlir::hec::DesignOp design, mlir::PatternRewriter &rewriter) {
    llvm::SmallVector<hec::PrimitiveOp, 4> toErase;
    for (unsigned i = 0; i < memories.size(); i++)
      if (memused[i] == 0) {
        toErase.push_back(memories[i].op);
      }
    for (auto op : toErase)
      op.erase();
  }
};

class Graph {
private:
  unsigned n;
  unsigned m;
  std::vector<std::vector<unsigned>> edges;
  std::vector<std::vector<unsigned>> backedges;
  std::vector<std::set<unsigned>> liveness;
  std::vector<std::set<unsigned>> def;
  std::vector<std::set<unsigned>> use;
  std::vector<unsigned> color;
  std::vector<std::vector<unsigned>> conflicts;

public:
  Graph(std::vector<TimeNode> &nodes) {
    n = nodes.size();
    m = 0;
    edges.resize(n);
    backedges.resize(n);
    for (auto node : nodes) {
      for (auto edge : node.edges) {
        edges[edge.from].push_back(edge.to);
        backedges[edge.to].push_back(edge.from);
      }
      if (node.type == TimeNode::NodeT::ENDFOR ||
          node.type == TimeNode::NodeT::ENDWHILE) {
        edges[node.tend].push_back(node.t);
        backedges[node.t].push_back(node.tend);
      }
    }
    def.resize(n);
    use.resize(n);
    liveness.resize(n);
  }
  void addDef(unsigned id, unsigned place) {
    m = id >= m ? id + 1 : m;
    def[place].insert(id);
  }
  void addUse(unsigned id, unsigned place) {
    m = id >= m ? id + 1 : m;
    use[place].insert(id);
  }
  bool same(std::set<unsigned> &old, std::set<unsigned> &newone) {
    if (old.size() != newone.size())
      return false;
    for (auto x : newone)
      if (!old.count(x))
        return false;
    return true;
  }
  void analysize() {
    // std::cerr << n << std::endl;
    for (unsigned i = 0; i < n; i++)
      liveness[i] = use[i];
    for (bool modified = true; modified;) {
      modified = false;
      for (unsigned i = n - 1; i != -1u; i--) {
        auto old_liveness = liveness[i];
        std::set<unsigned> new_liveness = use[i];
        for (auto succ : edges[i])
          for (auto succ_live : liveness[succ])
            if (!def[succ].count(succ_live))
              new_liveness.insert(succ_live);
        if (!same(old_liveness, new_liveness)) {
          liveness[i] = new_liveness;
          modified = true;
        }
      }
    }
    // for (unsigned i = 0; i < n; i++) {
    //   std::cerr << "Node " << i << " : ";
    //   std::cerr << "uses{";
    //   for (auto x : use[i])
    //     std::cerr << x << " ";
    //   std::cerr << "} defs{";
    //   for (auto x : def[i])
    //     std::cerr << x << " ";
    //   std::cerr << "} liveness{";
    //   for (auto x : liveness[i])
    //     std::cerr << x << " ";
    //   std::cerr << "}" << std::endl;
    // }
    conflicts.resize(m);
    for (unsigned i = 0; i < n; i++) {
      for (auto ptr = liveness[i].begin(); ptr != liveness[i].end(); ptr++)
        for (auto ptrj = ptr; ptrj != liveness[i].end(); ptrj++)
          if (ptrj != ptr) {
            conflicts[*ptr].push_back(*ptrj);
            conflicts[*ptrj].push_back(*ptr);
          }
    }
  }
  std::vector<unsigned> colorize() {
    color.resize(m);
    for (unsigned i = 0; i < m; i++)
      color[i] = -1; // means uncolored;

    for (unsigned i = 0; i < m; i++) {
      unsigned candidate = 0;
      std::set<unsigned> exists;
      for (auto x : conflicts[i])
        exists.insert(color[x]);
      while (exists.count(candidate))
        candidate += 1;
      color[i] = candidate;
    }

    std::cerr << "Coloring: ";
    for (auto x : color)
      std::cerr << x << " ";
    std::cerr << std::endl;
    return color;
  }
};

class STG {
private:
  enum Style { NORMAL, PIPELINEFOR, PIPELINEFUNC } style;
  std::map<std::string, hec::StateOp> str2State;
  size_t state_count = 0;
  std::map<size_t, size_t> node2State;

  std::vector<ValueUseDef> valueUseDefs;
  size_t value_count = 0;

  std::vector<Register> registers;
  size_t reg_count = 0;
  std::map<size_t, size_t> value2Reg;

  std::vector<Memory> memories;
  size_t mem_count = 0;
  std::map<size_t, size_t> value2Mem;

  std::vector<std::pair<mlir::Value, mlir::Value>> arg2arg;

  std::vector<mlir::Operation *> multiCycleOpCollection;
  std::map<mlir::Operation *, size_t> op2cell;
  std::vector<Cell> cells;
  size_t cell_count = 0;
  std::map<std::string, size_t> cellNameCount;

  std::vector<TimeNode> &nodes;
  mlir::tor::FuncOp func;
  GlobalStorage &glbStorage;

  std::map<std::string, llvm::SmallVector<mlir::hec::ComponentPortInfo, 4>>
      portDict;
  mlir::hec::ComponentOp component;
  mlir::hec::StateSetOp stg;
  mlir::hec::StageSetOp stageset;

  std::map<size_t, hec::InstanceOp> callDict;
  mlir::PatternRewriter &rewriter;

  std::vector<int> topo;

  struct Cycle {
    std::vector<unsigned> nodeIds, reviewIds;
    hec::StageOp stageop;
    Cycle() : stageop(nullptr) {
      nodeIds.clear();
      reviewIds.clear();
    }
  };
  std::vector<Cycle> cycles;
  std::vector<unsigned> depth;
  std::vector<std::map<unsigned, unsigned>> pipelineRegs;
  std::vector<std::vector<std::pair<mlir::Value, bool>>> guards;
  mlir::Value inductionWire;

private:
  size_t getClock(TimeEdge &edge) {
    auto dict = edge.attr.cast<mlir::DictionaryAttr>();
    auto format = dict.get("type").cast<StringAttr>().getValue();
    auto typestr = format.rsplit(":").first;
    auto clockstr = format.rsplit(":").second;
    unsigned long clock;
    if (typestr.equals("merge"))
      clock = 0ul;
    if (clockstr.size() == 0)
      clock = 1ul;
    else
      clock = std::stoul(clockstr.str());
    return std::max(1ul, clock);
  }

  unsigned set_cycle() {
    auto n = nodes.size();
    std::queue<int> q;
    depth.resize(n);
    std::fill(depth.begin(), depth.end(), 0);
    std::vector<size_t> degree(n);

    for (size_t i = 0; i < n; i++)
      degree[i] = nodes.at(i).backEdges.size();

    q.push(0);
    depth[0] = 0;
    if (style == Style::PIPELINEFOR)
      nodes[0].edges.pop_back();

    while (!q.empty()) {
      int x = q.front();
      q.pop();
      while (cycles.size() < 1ul + depth[x])
        cycles.push_back(Cycle());
      cycles[depth[x]].nodeIds.push_back(x);

      auto &node = nodes.at(x);
      for (auto edge : node.edges) {
        unsigned dep = depth[x] + getClock(edge);
        if (dep > depth[edge.to])
          depth[edge.to] = dep;
        degree[edge.to] -= 1;
        if (degree[edge.to] == 0)
          q.push(edge.to);
      }
    }

    std::cerr << "Depth: " << std::endl;
    for (size_t i = 0; i < n; i++)
      std::cerr << depth[i] << " ";
    std::cerr << std::endl;

    return cycles.size();
  }

  mlir::Value mapValue(mlir::Value val, int t = 0, std::string state = "") {
    // std::cerr << "map value ";
    // if (state != "")
    //   std::cerr << "on " << state << "; ";
    // else
    //   std::cerr << "; ";
    // val.dump();
    // std::cerr << std::endl;
    for (auto vud : valueUseDefs)
      if (vud.value == val) {
        switch (vud.type) {
        case ValueUseDef::Type::Constant:
          return vud.owner->getResults().front();
        case ValueUseDef::Type::Variable:
          if (vud.state != "") {
            // std::cerr << "The stored state is " << vud.state;
            // if (state != "")
            //   std::cerr << ", and queried in " << state << std::endl;
            // else
            //   std::cerr << std::endl;
          } else {
            // std::cerr << std::endl;
          }

          if (vud.state != "" && state != "" && state == vud.state &&
              vud.wire != nullptr)
            return vud.wire;
          else {
            // std::cerr << "vud.id = " << vud.id
            //           << ", rid = " << value2Reg[vud.id] << std::endl;
            // registers[value2Reg[vud.id]].op.dump();
            return registers[value2Reg[vud.id]].op.getResults().front();
          }
        case ValueUseDef::Type::Argument:
          if (t == 0) {
            for (auto pr : arg2arg)
              if (pr.first == val)
                return pr.second;
            return nullptr;
          } else
            return registers[value2Reg[vud.id]].op.getResults().front();
        case ValueUseDef::Type::GlobalVar:
          return glbStorage.getReg(vud.value);
        default:
          break;
        }
      }
    mlir::Value glbVal = glbStorage.getReg(val);
    if (glbVal != nullptr)
      return glbVal;
    glbVal = glbStorage.getConstant(val);
    if (glbVal != nullptr)
      return glbVal;

    return nullptr;
  }

  Memory mapValueMem(mlir::Value value) {
    std::cerr << "map value to mem: ";
    value.dump();
    for (auto vud : valueUseDefs)
      if (vud.value == value) {
        switch (vud.type) {
        case ValueUseDef::Type::Memory:
          return memories[value2Mem[vud.id]];
        case ValueUseDef::Type::GlobalMem:
          return glbStorage.getMem(vud.value);
        default:
          break;
        }
      }
    auto glbRes = glbStorage.getMem(value);
    if (glbRes.id == -1ul)
      assert(0 && "Memory Value not found");
    return glbRes;
  }

  mlir::Value mapOp2Reg(mlir::Operation *op) {
    // std::cerr << "map op to reg: " << std::endl;
    for (auto vud : valueUseDefs)
      if (vud.type == ValueUseDef::Type::NoVal && vud.owner == op)
        return registers[value2Reg[vud.id]].op.getResults().front();
    return nullptr;
  }

  Cell &getCellByOp(mlir::Operation *op) { return cells[op2cell[op]]; }
  bool valueInPipeline(mlir::Value value) {
    return false;
    bool need = 0;
    for (auto user : value.getUsers()) {
      auto ancient = user->getParentOp();
      while (!llvm::isa<tor::FuncOp>(ancient)) {
        if (auto forOp = llvm::dyn_cast<tor::ForOp>(ancient)) {
          auto pipelineAttr = forOp->getAttr("pipeline");
          if (pipelineAttr == nullptr)
            break;
          auto pipeline = pipelineAttr.cast<IntegerAttr>().getInt();
          if (pipeline)
            need = 1;
          break;
        } else
          ancient = ancient->getParentOp();
      }
    }
    return need;
  }

  void set_topo(unsigned t, unsigned tend, bool reach = 1) {
    // std::cerr << t << std::endl;
    if (t == tend) {
      if (reach)
        topo.push_back(t);
      return;
    } else
      topo.push_back(t);
    switch (nodes[t].type) {
    case TimeNode::NodeT::SEQ:
      set_topo(nodes[t].edges[0].to, tend, reach);
      break;
    case TimeNode::NodeT::CALL:
      set_topo(nodes[t].edges[0].to, tend, reach);
      break;
    case TimeNode::NodeT::IF:
      set_topo(nodes[t].edges[0].to, nodes[t].tend, 0);
      if (nodes[t].edges.size() == 2)
        set_topo(nodes[t].edges[1].to, nodes[t].tend, 0);
      set_topo(nodes[t].tend, tend, reach);
      break;
    case TimeNode::NodeT::WHILE:
      set_topo(nodes[t].edges[0].to, nodes[t].tend, reach);
      set_topo(nodes[t].edges[1].to, tend, reach);
      break;
    case TimeNode::NodeT::FOR:
      set_topo(nodes[t].edges[0].to, nodes[t].tend, reach);
      set_topo(nodes[t].edges[1].to, tend, reach);
      break;
    default:
      break;
    }
  }

  void gen_component(mlir::hec::DesignOp design) {
    std::cerr << "ComponentGen for FuncOp " << func.getName().str()
              << std::endl;

    auto context = design.getContext();
    rewriter.setInsertionPointToEnd(design.getBody());

    mlir::StringAttr name = func.getNameAttr();
    llvm::SmallVector<mlir::hec::ComponentPortInfo, 4> ports;
    mlir::StringAttr interfc =
        mlir::StringAttr::get(context, llvm::StringRef("naked"));
    mlir::StringAttr style =
        this->style == Style::NORMAL
            ? mlir::StringAttr::get(context, llvm::StringRef("STG"))
            : rewriter.getStringAttr("pipeline");

    auto funcType = func.getType();

    size_t icount = 0;
    for (auto inPort : funcType.getInputs()) {
      auto tmpstr = std::string("in") + std::to_string(icount++);
      llvm::StringRef port_name(tmpstr);

      ports.push_back(
          hec::ComponentPortInfo(mlir::StringAttr::get(context, port_name),
                                 inPort, hec::PortDirection::INPUT));

      std::cerr << "\t InPort  " << ports.back().name.getValue().str() << " : ";
      std::string typeStr;
      llvm::raw_string_ostream stro(typeStr);
      ports.back().type.print(stro);
      stro.flush();
      std::cerr << typeStr << std::endl;
    }

    icount += 1;
    ports.push_back(hec::ComponentPortInfo(mlir::StringAttr::get(context, "go"),
                                           mlir::IntegerType::get(context, 1),
                                           hec::PortDirection::INPUT));

    size_t ocount = 0;
    for (auto outPort : funcType.getResults()) {
      auto tmpstr = std::string("out") + std::to_string(ocount++);
      llvm::StringRef port_name(tmpstr);

      ports.push_back(
          hec::ComponentPortInfo(mlir::StringAttr::get(context, port_name),
                                 outPort, hec::PortDirection::OUTPUT));

      std::cerr << "\t OutPort  " << ports.back().name.getValue().str()
                << " : ";
      std::string typeStr;
      llvm::raw_string_ostream stro(typeStr);
      ports.back().type.print(stro);
      stro.flush();
      std::cerr << typeStr << std::endl;
    }

    ocount += 1;
    ports.push_back(hec::ComponentPortInfo(
        mlir::StringAttr::get(context, "done"),
        mlir::IntegerType::get(context, 1), hec::PortDirection::OUTPUT));

    component = rewriter.create<mlir::hec::ComponentOp>(design.getLoc(), name,
                                                        ports, interfc, style);

    switch (this->style) {
    case Style::NORMAL:
      stg = llvm::dyn_cast<hec::StateSetOp>(component.getBody()->back());
      break;
    case Style::PIPELINEFOR:
      component->setAttr("pipeline", rewriter.getStringAttr("for"));
      component->setAttr("II", func->getAttr("II"));
      this->stageset =
          llvm::dyn_cast<hec::StageSetOp>(component.getBody()->back());
      break;
    case Style::PIPELINEFUNC:
      component->setAttr("latency",
                         rewriter.getI32IntegerAttr(depth[nodes.size() - 1]));
      component->setAttr("pipeline", rewriter.getStringAttr("func"));
      if (func->getAttr("II") != nullptr)
        component->setAttr("II", func->getAttr("II"));
      this->stageset =
          llvm::dyn_cast<hec::StageSetOp>(component.getBody()->back());
      break;
    }
  }

  void set_regs() {
    value_count = 0;
    reg_count = 0;
    auto createVUD = [&](mlir::Value val, mlir::Operation *op, size_t id) {
      value_count++;

      ValueUseDef vud(value_count - 1, val, op, id,
                      ValueUseDef::Type::Variable);

      for (auto user : val.getUsers()) {
        // user->dump();
        if (auto ifop = llvm::dyn_cast<tor::IfOp>(user)) {
          if (this->style == Style::NORMAL)
            vud.use.push_back(ifop.starttime());
          else
            vud.use.push_back(ifop.endtime());
        } else if (auto yieldOp = llvm::dyn_cast<tor::YieldOp>(user)) {
          vud.use.push_back((yieldOp->getParentOp())
                                ->getAttr("endtime")
                                .template cast<::mlir::IntegerAttr>()
                                .getInt());
        } else if (auto returnOp = llvm::dyn_cast<tor::ReturnOp>(user)) {
          vud.use.push_back(nodes.size() - 1);
        } else if (auto conditionOp = llvm::dyn_cast<tor::ConditionOp>(user)) {
          std::cerr << "Meet a conditionOp" << std::endl;
        } else {
          vud.use.push_back(user->getAttr("starttime")
                                .template cast<::mlir::IntegerAttr>()
                                .getInt());
        }
      }

      // bool needPipeline = valueInPipeline(val);
      // if (needPipeline) {
      //   vud.type = ValueUseDef::Type::GlobalVar;
      //   glbStorage.add_reg(vud);
      // }

      valueUseDefs.push_back(vud);
    };

    // Registers storing arguments of the component
    auto argptr = component.getArguments().begin();
    for (auto arg : func.getArguments()) {
      value_count += 1;
      ValueUseDef vud(value_count - 1, arg, func, -2,
                      ValueUseDef::Type::Argument);

      // for (auto user : arg.getUsers()) {
      //   // user->dump();
      //   if (auto ifop = llvm::dyn_cast<tor::IfOp>(user)) {
      //     if (this->style == Style::NORMAL)
      //       vud.use.push_back(ifop.starttime());
      //     else
      //       vud.use.push_back(ifop.endtime());
      //   } else if (auto yieldOp = llvm::dyn_cast<tor::YieldOp>(user)) {
      //     vud.use.push_back((yieldOp->getParentOp())
      //                           ->getAttr("endtime")
      //                           .template cast<::mlir::IntegerAttr>()
      //                           .getInt());
      //   } else if (auto returnOp = llvm::dyn_cast<tor::ReturnOp>(user)) {
      //     vud.use.push_back(nodes.size() - 1);
      //   } else if (auto conditionOp =
      //   llvm::dyn_cast<tor::ConditionOp>(user)) {
      //     std::cerr << "Meet a conditionOp" << std::endl;
      //   } else {
      //     vud.use.push_back(user->getAttr("starttime")
      //                           .template cast<::mlir::IntegerAttr>()
      //                           .getInt());
      //   }
      // }

      arg2arg.push_back(std::make_pair(arg, *(argptr++)));

      if (style == Style::NORMAL && nodes.size() > 1) {
        reg_count += 1;
        Register reg(reg_count - 1);
        registers.push_back(reg);
        value2Reg[vud.id] = reg_count - 1;
      } else {

        for (auto user : arg.getUsers()) {
          // user->dump();
          if (auto ifop = llvm::dyn_cast<tor::IfOp>(user)) {
            vud.use.push_back(ifop.endtime());
          } else if (auto yieldOp = llvm::dyn_cast<tor::YieldOp>(user)) {
            vud.use.push_back((yieldOp->getParentOp())
                                  ->getAttr("endtime")
                                  .template cast<::mlir::IntegerAttr>()
                                  .getInt());
          } else if (auto returnOp = llvm::dyn_cast<tor::ReturnOp>(user)) {
            vud.use.push_back(nodes.size() - 1);
          } else if (auto conditionOp =
                         llvm::dyn_cast<tor::ConditionOp>(user)) {
            std::cerr << "Meet a conditionOp" << std::endl;
          } else {
            vud.use.push_back(user->getAttr("starttime")
                                  .template cast<::mlir::IntegerAttr>()
                                  .getInt());
          }
        }
      }
      valueUseDefs.push_back(vud);
    }

    for (auto &op : func.getBody().front())
      if (auto constant = llvm::dyn_cast<ConstantOp>(op)) {
        value_count += 1;
        ValueUseDef vud(value_count, constant.getResult(), constant, -1,
                        ValueUseDef::Type::Constant);
        valueUseDefs.push_back(vud);
      }

    for (size_t i = 0; i < nodes.size(); i++) {
      auto &node = nodes.at(topo[i]);
      switch (node.type) {
      case TimeNode::NodeT::SEQ:
        for (auto opOnEdge : node.opsOnEdge) {
          auto op = opOnEdge.op;
          for (auto res : op->getResults())
            createVUD(res, op,
                      op->getAttrOfType<IntegerAttr>("endtime").getInt());
        }
        break;

      case TimeNode::NodeT::CALL:
        if (auto callop =
                llvm::dyn_cast<tor::CallOp>(node.opsOnEdge.front().op)) {
          for (auto res : callop.getResults())
            createVUD(res, callop, callop.endtime());
        }
        break;

      case TimeNode::NodeT::IF:
        if (auto ifop = llvm::dyn_cast<tor::IfOp>(node.op)) {
          for (auto res : ifop.getResults()) {
            createVUD(res, ifop, ifop.endtime());
          }
        }
        for (auto opOnEdge : node.opsOnEdge) {
          auto op = opOnEdge.op;
          for (auto res : opOnEdge.op->getResults())
            createVUD(res, op,
                      op->getAttrOfType<IntegerAttr>("endtime").getInt());
        }
        break;
      case TimeNode::NodeT::WHILE:

        if (auto whileop = llvm::dyn_cast<tor::WhileOp>(node.op)) {
          valueUseDefs.push_back({value_count++,
                                  whileop.getRegion(0).getArgument(0), whileop,
                                  i, ValueUseDef::Type::Variable});
          valueUseDefs.back().use.push_back(whileop.endtime());

          for (auto arg : whileop.getRegion(1).getArguments()) {
            createVUD(arg, whileop, i);
          }

          for (auto res : whileop.getResults())
            createVUD(res, whileop, whileop.endtime());
        }
        for (auto opOnEdge : node.opsOnEdge) {
          auto op = opOnEdge.op;
          for (auto res : op->getResults())
            createVUD(res, op,
                      op->getAttrOfType<IntegerAttr>("endtime").getInt());
        }

        break;
      case TimeNode::NodeT::FOR:
        if (auto forop = llvm::dyn_cast<tor::ForOp>(node.op)) {
          auto II = node.getPipelineII();
          if (II == -1) {
            valueUseDefs.push_back(
                {value_count++, nullptr, forop, i, ValueUseDef::Type::NoVal});
            valueUseDefs.back().use.push_back(forop.endtime());
          }

          if (style == Style::NORMAL) {
            valueUseDefs.push_back({value_count++, forop.getInductionVar(),
                                    forop, i, ValueUseDef::Type::Variable});

            valueUseDefs.back().use.push_back(forop.endtime());
          } else {
            createVUD(forop.getInductionVar(), forop, i);
            valueUseDefs.back().use.push_back(forop.endtime());
          }
          for (auto arg : forop.getRegionIterArgs())
            createVUD(arg, forop, i);

          for (auto res : forop.results())
            createVUD(res, forop, forop.endtime());

          for (auto opOnEdge : node.opsOnEdge) {
            auto op = opOnEdge.op;
            for (auto res : op->getResults())
              createVUD(res, op,
                        op->getAttrOfType<IntegerAttr>("endtime").getInt());
          }
        }
        break;
      default:
        break;
      }

      std::cerr << std::endl;
    }

    allocate_regs();
  }
  void allocate_regs() {
    if (style == Style::NORMAL) {
      std::vector<mlir::Type> types;
      types.clear();

      auto found = [&](mlir::Type &type) {
        for (auto &x : types)
          if (x == type)
            return true;
        return false;
      };
      for (unsigned i = 0, n = valueUseDefs.size(); i < n; i++) {
        auto vud = valueUseDefs.at(i);
        mlir::Type type;
        switch (vud.type) {
        case ValueUseDef::Type::Variable:
          type = vud.value.getType();
          break;
        case ValueUseDef::Type::NoVal:
          type = rewriter.getIntegerType(1);
          break;
        default:
          continue;
        }
        if (!found(type)) {
          types.push_back(type);
          std::vector<size_t> vudsSameType;
          Graph dataflow(nodes);
          for (unsigned j = i; j < n; j++) {
            auto vud_ = valueUseDefs.at(j);
            if ((vud_.type == ValueUseDef::Type::NoVal &&
                 type == rewriter.getI1Type()) ||
                (vud_.type == ValueUseDef::Type::Variable &&
                 vud_.value.getType() == type)) {
              vudsSameType.push_back(j);
              auto id = vudsSameType.size() - 1;
              dataflow.addDef(id, vud_.def);
              for (auto use : vud_.use)
                dataflow.addUse(id, use);
            }
          }
          dataflow.analysize();
          std::vector<unsigned> coloring = dataflow.colorize();
          std::cerr << "Coloring: ";
          for (auto x : coloring)
            std::cerr << x << " ";
          std::cerr << std::endl;

          auto old_count = reg_count;
          unsigned num_new_cols =
              *std::max_element(coloring.begin(), coloring.end()) + 1;

          std::cerr << "# of new colors: " << num_new_cols << std::endl;

          for (unsigned j = 0; j < num_new_cols; j++)
            registers.push_back(Register(reg_count++, type));

          std::cerr << "old_count = " << old_count << std::endl;
          std::cerr << "new_count = " << reg_count << std::endl;
          for (unsigned j = 0, m = vudsSameType.size(); j < m; j++) {
            value2Reg[valueUseDefs[vudsSameType[j]].id] =
                old_count + coloring[j];
            std::cerr << "set value " << valueUseDefs[vudsSameType[j]].id
                      << " has reg " << old_count + coloring[j] << "\n";
          }
        }
      }
    } else if (style == Style::PIPELINEFOR || style == Style::PIPELINEFUNC) {
      pipelineRegs.resize(valueUseDefs.size());
      for (auto vud : valueUseDefs) {
        unsigned starttime, endtime = 0;
        if (vud.type == ValueUseDef::Type::Argument)
          starttime = 0;
        else if (vud.type == ValueUseDef::Type::Variable)
          starttime = depth[vud.def];
        else
          continue;

        if (auto ifop = llvm::dyn_cast<tor::IfOp>(vud.owner))
          starttime -= 1;

#define PROCESSCOMB(type)                                                      \
  if (auto sop = llvm::dyn_cast<type>(vud.owner))                              \
    starttime -= 1;
        PROCESSCOMB(tor::AddIOp)
        PROCESSCOMB(tor::SubIOp)
        PROCESSCOMB(tor::CmpIOp)
        PROCESSCOMB(AndOp)
        PROCESSCOMB(OrOp)
        PROCESSCOMB(XOrOp)
        PROCESSCOMB(ShiftLeftOp)
        PROCESSCOMB(SignedShiftRightOp)
        PROCESSCOMB(TruncateIOp)
        PROCESSCOMB(SignExtendIOp)
        PROCESSCOMB(NegFOp)
        PROCESSCOMB(SelectOp)
#undef PROCESSCOMB

        std::cerr << "Value ";
        vud.value.dump();
        for (auto use : vud.use) {
          std::cerr << "use at " << use << ", depth is " << depth[use]
                    << std::endl;

          endtime = (unsigned)depth[use] > endtime ? depth[use] : endtime;
        }

        if (vud.type == ValueUseDef::Type::Argument &&
            style == Style::PIPELINEFOR)
          endtime = 0;

        std::cerr << "starttime is " << starttime << ", endtime is " << endtime
                  << std::endl;

        pipelineRegs[vud.id].clear();

        unsigned lb, ub;
        if (style == Style::PIPELINEFOR) {
          lb = starttime + 1;
          ub = endtime + 1;
        } else {
          lb = starttime + 1;
          ub = endtime + 1;
        }
        for (unsigned i = lb; i <= ub; i++) {
          rewriter.setInsertionPoint(stageset);
          llvm::SmallVector<mlir::Type, 1> types;
          types.push_back(vud.value.getType());
          auto primitive = rewriter.create<hec::PrimitiveOp>(
              component.getLoc(), mlir::TypeRange(types),
              rewriter.getStringAttr(
                  std::string("r") + "_" + std::to_string(vud.id) +
                  (style == Style::PIPELINEFOR &&
                           vud.value == llvm::dyn_cast<tor::ForOp>(nodes[0].op)
                                            .getInductionVar()
                       ? "_i"
                       : "") +
                  "_" + std::to_string(i)),
              rewriter.getStringAttr("register"));
          Register reg(reg_count++, vud.value.getType(), primitive);
          registers.push_back(reg);
          pipelineRegs[vud.id][i] = reg.id;
        }
      }
    } else {
      assert(0 && "Undefined component style");
    }
  }

  void allocate_cells() {
    if (style == Style::NORMAL) {
      std::set<std::string> allocated;
      for (auto ptri = cells.begin(); ptri != cells.end(); ptri++)
        if (!allocated.count(ptri->pname)) {
          auto pname = ptri->pname;
          allocated.insert(pname);
          Graph dataflow(nodes);
          std::vector<unsigned> idList;
          for (auto ptrj = ptri; ptrj != cells.end(); ptrj++)
            if (ptrj->pname == pname) {
              idList.push_back(ptrj->id);
              unsigned id = idList.size() - 1;
              dataflow.addDef(id, ptrj->t);
              dataflow.addUse(id, ptrj->tend);
            }
          dataflow.analysize();
          std::vector<unsigned> coloring = dataflow.colorize();

          for (unsigned i = 0, m = idList.size(); i < m; i++) {
            cells[idList[i]].cname += "_" + std::to_string(coloring[i]);
            std::cerr << "set cell " << idList[i] << " has cname "
                      << cells[idList[i]].cname << std::endl;
          }
        }
    } else {
      std::map<std::string, unsigned> cname_count;
      std::map<std::string, std::set<unsigned>> cell_usage;

      unsigned II = component->getAttrOfType<IntegerAttr>("II").getInt();
      for (auto &cell : cells) {
        bool found = false;
        unsigned d = depth[cell.t] % II;
        for (unsigned i = 0; i < cname_count[cell.cname]; i++) {
          std::string cellname = cell.cname + "_" + std::to_string(i);
          if (cell_usage[cellname].count(d) == 0) {
            found = true;
            cell.cname = cellname;
            cell_usage[cellname].insert(d);
          }
        }
        if (!found) {
          cell.cname +=
              std::string("_") + std::to_string(cname_count[cell.cname]++);
          cell_usage[cell.cname].insert(d);
        }

        std::cerr << "bind op on node " << cell.t << " with depth " << d
                  << " to " << cell.cname << std::endl;
      }
      //   cell.cname += "_" + std::to_string(cname_count[cell.cname]++);
    }
  }

  void set_cells() {
    cell_count = 0;
    func.walk([&](mlir::Operation *op) {
#define addCell(OpType, cname, pname)                                          \
  if (auto top = llvm::dyn_cast<OpType>(op)) {                                 \
    multiCycleOpCollection.push_back(top);                                     \
    Cell cell(cell_count++, top.starttime(), top.endtime(),                    \
              std::string(cname) + "_" + func.getName().str(), pname);         \
    op2cell[top] = cell.id;                                                    \
    cell.setTypes(top);                                                        \
    cells.push_back(cell);                                                     \
  }
      addCell(tor::AddFOp, "addf", "add_float");
      addCell(tor::SubFOp, "subf", "sub_float");
      addCell(tor::MulIOp, "muli", "mul_integer");
      addCell(tor::MulFOp, "mulf", "mul_float");
      addCell(tor::DivFOp, "divf", "div_float");
#undef addCell

#define addSTDCell(OpType, cname, pname)                                       \
  if (auto top = llvm::dyn_cast<OpType>(op)) {                                 \
    multiCycleOpCollection.push_back(top);                                     \
    Cell cell(                                                                 \
        cell_count++, top->getAttrOfType<IntegerAttr>("starttime").getInt(),   \
        top->getAttrOfType<IntegerAttr>("endtime").getInt(), cname, pname);    \
    op2cell[top] = cell.id;                                                    \
    cell.setTypes(top);                                                        \
    cells.push_back(cell);                                                     \
  }

      addSTDCell(SIToFPOp, "i2f", "sitofp");
      addSTDCell(FPToSIOp, "f2i", "fptosi");
      addSTDCell(SignedDivIOp, "divi", "div_integer");
#undef addSTDCell

#define addCmpCell(OpType, cname, pname)                                       \
  if (auto top = llvm::dyn_cast<OpType>(op)) {                                 \
    multiCycleOpCollection.push_back(top);                                     \
    Cell cell(cell_count++, top.starttime(), top.endtime(),                    \
              std::string(cname) + "_" + func.getName().str(),                 \
              std::string(pname) + "_" +                                       \
                  tor::stringifyEnum(top.predicate()).str());                 \
    op2cell[top] = cell.id;                                                    \
    cell.setTypes(top);                                                        \
    cells.push_back(cell);                                                     \
  }
      addCmpCell(tor::CmpFOp, "cmpf", "cmp_float");
#undef addCmpCell
    });

    allocate_cells();
  }

  void gen_cells() {
    std::map<std::string, hec::PrimitiveOp> created;
    for (auto &cell : cells)
      if (!created.count(cell.cname)) {
        if (this->style == Style::NORMAL)
          rewriter.setInsertionPoint(stg);
        else
          rewriter.setInsertionPoint(stageset);
        cell.primitive = rewriter.create<hec::PrimitiveOp>(
            component.getLoc(), TypeRange(cell.types), cell.cname, cell.pname);
        created[cell.cname] = cell.primitive;
      } else
        cell.primitive = created[cell.cname];
  }

  void gen_mems() {
    func.walk([&](mlir::Operation *op) {
      if (auto alloc = llvm::dyn_cast<tor::AllocOp>(op)) {
        auto value = alloc.getResult();
        ValueUseDef vud(value_count++, value, alloc, -1,
                        ValueUseDef::Type::Memory);
        bool needPipeline = valueInPipeline(value);

        if (needPipeline) {
          vud.type = ValueUseDef::Type::GlobalMem;
          glbStorage.add_mem(alloc);
          valueUseDefs.push_back(vud);
        } else {
          valueUseDefs.push_back(vud);

          Memory mem(mem_count++);
          auto mem_type = alloc.getType();
          auto rw = mem_type.getRw();
          auto shape = mem_type.getShape();
          assert(shape.size() == 1 && "Only support 1-dim memory");
          auto length = shape.front();
          auto addressLen = static_cast<unsigned long>(ceil(log2(length)));
          auto elementType = mem_type.getElementType();
          llvm::SmallVector<mlir::Type> portTypes;
          if (rw.getValue() == "r") {
            portTypes.push_back(rewriter.getI1Type());
            portTypes.push_back(rewriter.getIntegerType(addressLen));
            portTypes.push_back(elementType);
            mem.type = Memory::Type::R;
          } else if (rw.getValue() == "w") {
            portTypes.push_back(rewriter.getI1Type());
            portTypes.push_back(rewriter.getI1Type());
            portTypes.push_back(rewriter.getIntegerType(addressLen));
            portTypes.push_back(elementType);
            mem.type = Memory::Type::W;
          } else if (rw.getValue() == "rw") {
            portTypes.push_back(rewriter.getI1Type());
            portTypes.push_back(rewriter.getI1Type());
            portTypes.push_back(rewriter.getIntegerType(addressLen));
            portTypes.push_back(elementType);
            portTypes.push_back(elementType);
            mem.type = Memory::Type::RW;
          }
          auto cname = rewriter.getStringAttr(std::string("mem_") +
                                              component.getName().str() + +"_" +
                                              std::to_string(mem_count - 1));
          auto pname = rewriter.getStringAttr(
              std::string("mem") /*+ "_" + rw.getValue().str()*/);

          rewriter.setInsertionPoint(stg);

          auto primitive = rewriter.create<hec::PrimitiveOp>(
              component.getLoc(),
              llvm::ArrayRef<mlir::Type>(portTypes.begin(), portTypes.end()),
              cname, pname);
          primitive->setAttr("len", rewriter.getI32IntegerAttr(length));
          primitive->setAttr("ports", rw);

          mem.op = primitive;
          memories.push_back(mem);
          value2Mem[vud.id] = mem.id;
        }
      }
    });
  }

  void gen_regs() {
    std::cerr << "Generate Registers ----" << std::endl;
    std::set<unsigned> created;
    for (auto vud : valueUseDefs)
      if (vud.type != ValueUseDef::Type::Constant &&
          !(vud.type == ValueUseDef::Type::Argument && nodes.size() < 2) &&
          vud.type != ValueUseDef::Type::GlobalVar &&
          vud.type != ValueUseDef::Type::GlobalMem) {
        std::string varName;
        llvm::raw_string_ostream stro(varName);
        if (vud.value != nullptr) {
          vud.value.print(stro);
          std::cerr << "Create register for " << varName
                    << ", reg_id = " << value2Reg[vud.id] << std::endl;
        }

        if (created.count(value2Reg[vud.id])) {
          std::cerr << "Already created, continue" << std::endl;
          continue;
        }
        created.insert(value2Reg[vud.id]);

        rewriter.setInsertionPoint(&component.getBody()->back());

        auto context = component.getContext();
        llvm::SmallVector<mlir::Type, 2> types;
        if (vud.value == nullptr)
          types.push_back(rewriter.getI1Type());
        else
          types.push_back(vud.value.getType());
        auto instanceName = mlir::StringAttr::get(
            context,
            llvm::StringRef(std::string("r_") + component.getName().str() +
                            "_" + std::to_string(value2Reg[vud.id])));
        auto primitiveName = mlir::StringAttr::get(context, "register");
        auto primitive = rewriter.create<mlir::hec::PrimitiveOp>(
            component.getLoc(), mlir::TypeRange(types), instanceName,
            primitiveName);
        registers[value2Reg[vud.id]].op = primitive;
      }
  }

  void gen_calls() {
    std::cerr << "Generate Calls ----" << std::endl;

    std::map<std::string, unsigned long> nameOccupy;

    for (auto node : nodes) {
      if (node.type == TimeNode::NodeT::CALL) {
        auto call = llvm::dyn_cast<tor::CallOp>(node.op);

        auto callee = call.callee();

        assert(stg != nullptr && "STG must be created first");
        rewriter.setInsertionPoint(stg);

        int id = nameOccupy[callee.str()]++;
        std::string instanceName = callee.str() + "_" + std::to_string(id);

        auto ports = portDict[callee.str()];

        std::cerr << "\t" << callee.str() << " : ";
        for (auto port : ports)
          std::cerr << port.name.getValue().str() << " ";
        std::cerr << std::endl;

        llvm::SmallVector<mlir::Type, 4> types;
        for (auto port : ports)
          types.push_back(port.type);

        auto instance = rewriter.create<hec::InstanceOp>(
            component.getLoc(), mlir::TypeRange(types), instanceName, callee);
        callDict[node.t] = instance;
      }
    }
  }

  ValueUseDef &getVUD(mlir::Value value, mlir::Operation *op) {
    for (auto &vud : valueUseDefs)
      if (vud.value == value ||
          (value == nullptr && vud.type == ValueUseDef::Type::NoVal &&
           vud.owner == op))
        return vud;
    assert(0);
  }
  void set_state_seq(size_t t, std::string state0, std::string state1) {
    auto node = nodes.at(t);
    for (auto opoe : node.opsOnEdge)
      for (auto res : opoe.op->getResults()) {
        auto &vud = getVUD(res, opoe.op);

        bool isComb = false;
#define CHECKCOMB(OpType)                                                      \
  if (auto sop = llvm::dyn_cast<OpType>(vud.owner))                            \
    isComb = true;
        CHECKCOMB(tor::AddIOp)
        CHECKCOMB(tor::SubIOp)
        CHECKCOMB(tor::CmpIOp)
        CHECKCOMB(AndOp)
        CHECKCOMB(OrOp)
        CHECKCOMB(XOrOp)
        CHECKCOMB(ShiftLeftOp)
        CHECKCOMB(SignedShiftRightOp)
        CHECKCOMB(TruncateIOp)
        CHECKCOMB(SignExtendIOp)
        CHECKCOMB(NegFOp)
        CHECKCOMB(SelectOp)
#undef CHECKCOMB
        vud.state = isComb ? state0 : state1;
      }
  }

  void set_state_call(size_t t, std::string state) {
    auto node = nodes.at(t);
    for (auto opoe : node.opsOnEdge)
      for (auto res : opoe.op->getResults()) {
        auto &vud = getVUD(res, opoe.op);
        vud.state = state;
      }
  }

  void set_state_if(size_t t, std::string state) {
    auto node = nodes.at(t);
    for (auto opoe : node.opsOnEdge)
      for (auto res : opoe.op->getResults()) {
        auto &vud = getVUD(res, opoe.op);
        vud.state = state;
      }
  }

  void set_state_while(size_t t, std::string state, tor::WhileOp whileop) {
    auto node = nodes.at(t);
    for (auto opoe : node.opsOnEdge)
      for (auto res : opoe.op->getResults()) {
        auto &vud = getVUD(res, opoe.op);
        vud.state = state;
      }

    auto ptr = whileop.getOperands().begin();
    auto &cond_vud = getVUD(whileop.getRegion(0).getArgument(0), whileop);
    cond_vud.state = state;
    cond_vud.wire = mapValue(*ptr++, t, state);

    for (auto arg : whileop.getRegion(1).getArguments()) {
      auto &vud = getVUD(arg, whileop);
      vud.state = state;
      vud.wire = mapValue(*ptr++, t, state);
    }
  }

  void set_state_for(size_t t, std::string state, tor::ForOp forop) {
    auto node = nodes.at(t);
    for (auto opoe : node.opsOnEdge)
      for (auto res : opoe.op->getResults()) {
        auto &vud = getVUD(res, opoe.op);
        vud.state = state;
      }

    auto &i_vud = getVUD(forop.getInductionVar(), forop);
    i_vud.state = state;
    i_vud.wire = mapValue(forop.lowerBound(), t, state);

    auto ptr = forop.getIterOperands().begin();
    for (auto arg : forop.getRegionIterArgs()) {
      auto &vud = getVUD(arg, forop);
      vud.state = state;
      vud.wire = mapValue(*ptr++, t, state);
    }
  }

  size_t gen_states(size_t t, size_t tend, bool reach = 1) {
    auto getClock = [](TimeEdge &te) {
      auto dict = te.attr.cast<mlir::DictionaryAttr>();
      auto format = dict.get("type").cast<StringAttr>().getValue();
      auto typestr = format.rsplit(":").first;
      auto clockstr = format.rsplit(":").second;
      unsigned long clock;
      if (typestr.equals("merge"))
        clock = 0ul;
      else if (clockstr.size() == 0)
        clock = 1ul;
      else
        clock = std::stoul(clockstr.str());
      return clock;
    };

    auto createState = [&](std::string sno) {
      rewriter.setInsertionPointToEnd(stg.getBody());
      auto context = stg.getContext();
      auto name = mlir::StringAttr::get(context, std::string("s") + sno);
      auto initial = mlir::IntegerAttr::get(mlir::IntegerType::get(context, 1),
                                            sno == "0");
      auto stateOp = rewriter.create<hec::StateOp>(stg.getLoc(), name, initial);
      rewriter.setInsertionPointToEnd(stateOp.getBody());
      auto transOp = rewriter.create<hec::TransitionOp>(stateOp.getLoc());
      transOp.getRegion().push_back(new mlir::Block());

      str2State[sno] = stateOp;

      return stateOp;
    };

    auto createStateAfter = [&](std::string sno, mlir::Operation *op) {
      rewriter.setInsertionPointAfter(op);
      auto context = stg.getContext();
      auto name = mlir::StringAttr::get(context, std::string("s") + sno);
      auto initial =
          mlir::IntegerAttr::get(mlir::IntegerType::get(context, 1), 0);

      auto state = rewriter.create<hec::StateOp>(stg.getLoc(), name, initial);

      rewriter.setInsertionPointToEnd(state.getBody());
      auto trans = rewriter.create<hec::TransitionOp>(state.getLoc());
      trans.getRegion().push_back(new mlir::Block());

      str2State[sno] = state;
      return state;
    };

    auto createGoto = [&](hec::StateOp state, std::string nextname) {
      auto trans = llvm::dyn_cast<hec::TransitionOp>(state.getBody()->back());
      rewriter.setInsertionPointToEnd(trans.getBody());
      auto next =
          mlir::StringAttr::get(stg.getContext(), std::string("s") + nextname);
      rewriter.create<hec::GotoOp>(trans.getLoc(), next.getValue(), nullptr);
    };

    auto createGotoCond = [&](hec::StateOp state, std::string nextname,
                              mlir::Value cond) {
      auto trans = llvm::dyn_cast<hec::TransitionOp>(state.getBody()->back());
      rewriter.setInsertionPointToEnd(trans.getBody());
      auto next =
          mlir::StringAttr::get(stg.getContext(), std::string("s") + nextname);
      rewriter.create<hec::GotoOp>(trans.getLoc(), next.getValue(), cond);
    };

    std::cerr << "Now is node " << t << ", tend is " << tend << ", reach is "
              << reach << std::endl;
    if (reach == 0 && nodes[t].edges.size() == 1 &&
        nodes[t].edges[0].to == tend) {
      state_count += 1;
      node2State[t] = state_count - 1;
      str2State[std::to_string(state_count - 1)] =
          createState(std::to_string(state_count - 1));
      return t;
    }
    if (tend == -1ul ||
        ((nodes[t].edges.size() == 0 || t == nodes.size() - 1) &&
         node2State.find(t) == node2State.end())) {
      state_count += 1;
      node2State[t] = state_count - 1;

      str2State[std::to_string(state_count - 1)] =
          createState(std::to_string(state_count - 1));

      return t;
    }

    if (t == tend || nodes[t].edges.size() == 0 || t == nodes.size() - 1)
      return t;

    auto &node = nodes[t];
    auto id = state_count++;
    size_t ret = 0;

    std::cerr << "StateGen for node " << t << ": ";
    switch (node.type) {
    case TimeNode::NodeT::SEQ:
      std::cerr << "SEQ" << std::endl;
      break;
    case TimeNode::NodeT::CALL:
      std::cerr << "CALL" << std::endl;
      break;
    case TimeNode::NodeT::IF:
      std::cerr << "IF" << std::endl;
      break;
    case TimeNode::NodeT::FOR:
      std::cerr << "FOR" << std::endl;
      break;
    case TimeNode::NodeT::WHILE:
      std::cerr << "WHILE" << std::endl;
      break;
    default:
      break;
    }
    if (node.type == TimeNode::NodeT::SEQ) {
      // SEQGen: State generation of SEQ
      auto state0 = createState(std::to_string(id));

      node2State[t] = id;

      assert(node.edges.size() == 1 && "SEQ node has one successor");

      auto clock = getClock(node.edges[0]);

      // if (clock == 0) return t;
      if (node.edges[0].to == tend && !reach)
        return t;

      ret = gen_states(node.edges[0].to, tend, reach);

      std::vector<hec::StateOp> statebags;
      statebags.push_back(state0);

      if (clock == 0)
        clock = 1;

      for (unsigned long i = 0; i < clock - 1; i++) {
        auto curState = statebags[i];
        auto nextState = createStateAfter(
            std::to_string(id) + std::string("_") + std::to_string(i),
            curState);
        statebags.push_back(nextState);

        createGoto(curState,
                   std::to_string(id) + std::string("_") + std::to_string(i));
      }

      unsigned long nextNodeId;
      nextNodeId = nodes[t].edges[0].to;

      std::string nextname = std::to_string(node2State[nextNodeId]);

      createGoto(statebags.back(), nextname);

      set_state_seq(t, state0.getName().str(), std::string("s") + nextname);
    } else if (node.type == TimeNode::NodeT::CALL) {
      // CALLGen
      auto state0 = createState(std::to_string(id));
      node2State[t] = id;

      assert(node.edges.size() == 1 && "CALL node has one successor");
      assert(node.opsOnEdge.size() == 1 && "CALL node has one op");

      ret = gen_states(node.edges[0].to, tend, reach);

      auto statewait = createStateAfter(std::to_string(id) + "_wait", state0);

      unsigned long long nextNodeId;

      nextNodeId = nodes[t].edges[0].to;
      std::string nextState = std::to_string(node2State[nextNodeId]);

      auto call = callDict[t];
      createGoto(state0, std::to_string(id) + "_wait");
      createGotoCond(state0, nextState, call.getResults().back());
      createGotoCond(statewait, nextState, call.getResults().back());

      set_state_call(t, state0.getName().str());
    } else if (node.type == TimeNode::NodeT::IF) {
      // IFGen
      auto state0 = createState(std::to_string(id));
      node2State[t] = id;

      assert(node.edges.size() >= 1 && node.edges.size() <= 2 &&
             "IF nodes has at one or two successor");

      auto ifop = llvm::dyn_cast<tor::IfOp>(node.op);
      if (node.edges.size() == 2) {
        node.tthen = gen_states(node.edges[0].to, node.tend, 0);
        node.telse = gen_states(node.edges[1].to, node.tend, 0);

        std::cerr << "then end is " << node.tthen << ", "
                  << "else end is " << node.telse << std::endl;

        ret = gen_states(node.tend, tend, reach);

        rewriter.setInsertionPoint(&state0.getBody()->back());
        auto notcond =
            rewriter.create<hec::NotOp>(state0.getLoc(), rewriter.getI1Type(),
                                        mapValue(ifop.condition(), t), nullptr);

        node.val = notcond.res();

        if (node.edges[0].to != node.tend) {
          auto clock = getClock(node.edges[0]);
          assert(clock >= 1);
          std::vector<hec::StateOp> statebags;
          statebags.push_back(state0);
          for (unsigned long i = 0; i < clock - 1; i++) {
            auto curState = statebags[i];
            auto nextState = createStateAfter(
                std::to_string(id) + std::string("_then_") + std::to_string(i),
                curState);
            statebags.push_back(nextState);

            if (i)
              createGoto(curState, std::to_string(id) + std::string("_then_") +
                                       std::to_string(i));
            else
              createGotoCond(curState,
                             std::to_string(id) + std::string("_then_") +
                                 std::to_string(i),
                             mapValue(ifop.condition(), t));
          }
          auto nextNodeId = node.edges[0].to;
          std::string nextname = std::to_string(node2State[nextNodeId]);

          if (statebags.size() > 1)
            createGoto(statebags.back(), nextname);
          else
            createGotoCond(statebags.back(), nextname,
                           mapValue(ifop.condition(), t));
        } else {
          createGotoCond(state0, std::to_string(node2State[node.edges[0].to]),
                         mapValue(ifop.condition(), t));
        }

        if (node.edges[1].to != node.tend) {
          auto clock = getClock(node.edges[1]);
          assert(clock >= 1);
          std::vector<hec::StateOp> statebags;
          statebags.push_back(state0);
          for (unsigned long i = 0; i < clock - 1; i++) {
            auto curState = statebags[i];
            auto nextState = createStateAfter(
                std::to_string(id) + std::string("_else_") + std::to_string(i),
                curState);
            statebags.push_back(nextState);

            if (i)
              createGoto(curState, std::to_string(id) + std::string("_else_") +
                                       std::to_string(i));
            else
              createGotoCond(curState,
                             std::to_string(id) + std::string("_else_") +
                                 std::to_string(i),
                             node.val);
          }
          auto nextNodeId = node.edges[1].to;
          std::string nextname = std::to_string(node2State[nextNodeId]);
          if (statebags.size() > 1)
            createGoto(statebags.back(), nextname);
          else
            createGotoCond(statebags.back(), nextname, node.val);
        } else {
          createGotoCond(state0, std::to_string(node2State[node.edges[1].to]),
                         node.val);
        }

        if (node.tthen != node.tend) {
          auto stateid = node2State[node.tthen];
          auto stateThenEnd = str2State[std::to_string(stateid)];
          createGoto(stateThenEnd, std::to_string(node2State[node.tend]));
        }

        if (node.telse != node.tend) {
          auto stateid = node2State[node.telse];
          auto stateElseEnd = str2State[std::to_string(stateid)];
          createGoto(stateElseEnd, std::to_string(node2State[node.tend]));
        }
      } else {
        node.tthen = gen_states(node.edges[0].to, node.tend, 0);
        gen_states(node.tend, tend, reach);
        rewriter.setInsertionPoint(&state0.getBody()->back());
        auto notcond =
            rewriter.create<hec::NotOp>(state0.getLoc(), rewriter.getI1Type(),
                                        mapValue(ifop.condition(), t), nullptr);

        node.val = notcond.res();
        if (node.edges[0].to != node.tend) {
          auto clock = getClock(node.edges[0]);
          assert(clock >= 1);
          std::vector<hec::StateOp> statebags;
          statebags.push_back(state0);
          for (unsigned long i = 0; i < clock - 1; i++) {
            auto curState = statebags[i];
            auto nextState = createStateAfter(
                std::to_string(id) + std::string("_then_") + std::to_string(i),
                curState);
            statebags.push_back(nextState);

            if (i)
              createGoto(curState, std::to_string(id) + std::string("_then_") +
                                       std::to_string(i));
            else
              createGotoCond(curState,
                             std::to_string(id) + std::string("_then_") +
                                 std::to_string(i),
                             mapValue(ifop.condition(), t));
          }
          auto nextNodeId = node.edges[0].to;
          std::string nextname = std::to_string(node2State[nextNodeId]);
          if (statebags.size() > 1)
            createGoto(statebags.back(), nextname);
          else
            createGotoCond(statebags.back(), nextname,
                           mapValue(ifop.condition(), t));
        } else {
          createGotoCond(state0, std::to_string(node2State[node.edges[0].to]),
                         mapValue(ifop.condition(), t));
        }

        if (node.tthen != node.tend) {
          auto stateid = node2State[node.tthen];
          auto stateThenEnd = str2State[std::to_string(stateid)];
          createGoto(stateThenEnd, std::to_string(node2State[node.tend]));
        }

        createGotoCond(state0, std::to_string(node2State[node.tend]), node.val);
      }

      set_state_if(t, state0.getName().str());
    } else if (node.type == TimeNode::NodeT::WHILE) {
      auto state0 = createState(std::to_string(id));
      auto state1 = createStateAfter(std::to_string(id) + "_entry", state0);
      node2State[t] = id;
      auto whileop = llvm::dyn_cast<tor::WhileOp>(node.op);

      rewriter.setInsertionPoint(&state0.getBody()->back());
      auto notcond0 = rewriter.create<hec::NotOp>(
          state0.getLoc(), rewriter.getI1Type(),
          mapValue(whileop.getOperand(0), t), nullptr);
      node.val0 = notcond0.res();

      rewriter.setInsertionPoint(&state1.getBody()->back());
      auto notcond1 = rewriter.create<hec::NotOp>(
          state1.getLoc(), rewriter.getI1Type(),
          mapValue(whileop.getRegion(0).getArgument(0), t), nullptr);
      node.val1 = notcond1.res();

      auto while_end = node.edges[1].to;
      assert(node.edges.size() == 2 && "WHILE Node must have two successors");

      auto do_end = gen_states(node.edges[0].to, while_end, reach);

      node.tthen = do_end;
      std::cerr << "while_end is " << do_end
                << ", state_id = " << node2State[do_end] << std::endl;
      ret = gen_states(while_end, tend, reach);

      auto clock = getClock(node.edges[0]);

      std::vector<hec::StateOp> bags0;
      std::vector<hec::StateOp> bags1;
      bags0.push_back(state0);
      bags1.push_back(state1);

      assert(clock >= 1);

      for (unsigned long i = 0; i < clock - 1; i++) {
        auto curState = bags0[i];
        auto nextState = createStateAfter(
            std::to_string(id) + std::string("_") + std::to_string(i),
            curState);
        bags0.push_back(nextState);

        if (i)
          createGoto(curState,
                     std::to_string(id) + std::string("_") + std::to_string(i));
        else
          createGotoCond(curState,
                         std::to_string(id) + std::string("_") +
                             std::to_string(i),
                         mapValue(node.op->getOperand(0), t));
        curState = bags1[i];
        nextState = createStateAfter(
            std::to_string(id) + std::string("_entry_") + std::to_string(i),
            curState);
        bags1.push_back(nextState);

        if (i)
          createGoto(curState, std::to_string(id) + std::string("_entry_") +
                                   std::to_string(i));
        else
          createGotoCond(curState,
                         std::to_string(id) + std::string("_entry_") +
                             std::to_string(i),
                         mapValue(llvm::dyn_cast<tor::WhileOp>(node.op)
                                      .getRegion(0)
                                      .getArgument(0),
                                  t));
      }

      unsigned long nextNodeId;
      nextNodeId = node.edges[0].to;
      std::string nextname = std::to_string(node2State[nextNodeId]);
      createGoto(bags0.back(), nextname);
      createGoto(bags1.back(), nextname);

      createGotoCond(state0, std::to_string(node2State[while_end]),
                     notcond0.res());
      createGotoCond(state1, std::to_string(node2State[while_end]),
                     notcond1.res());

      createGoto(str2State[std::to_string(node2State[do_end])],
                 std::to_string(id) + std::string("_entry"));

      set_state_while(t, state0.getName().str(), whileop);
    } else if (node.type == TimeNode::NodeT::FOR) {
      auto state0 = createState(std::to_string(id));
      auto state1 = createStateAfter(std::to_string(id) + "_entry", state0);
      node2State[t] = id;
      auto forop = llvm::dyn_cast<tor::ForOp>(node.op);

      rewriter.setInsertionPoint(&state0.getBody()->back());
      auto cond = rewriter.create<hec::CmpIOp>(
          state0.getLoc(), rewriter.getI1Type(),
          // mapValue(forop.getInductionVar(), t),
          mapValue(forop.lowerBound(), t), mapValue(forop.upperBound(), t),
          rewriter.getStringAttr("sle"), nullptr);
      node.val0 = cond.res();

      rewriter.setInsertionPointAfter(cond);

      node.val0.dump();
      auto val = mapOp2Reg(forop);
      val.dump();

      auto assign = rewriter.create<hec::AssignOp>(
          state0.getLoc(), mapOp2Reg(forop), node.val0, nullptr);

      rewriter.setInsertionPointAfter(assign);
      auto notcond = rewriter.create<hec::NotOp>(
          state0.getLoc(), rewriter.getI1Type(), cond.res(), nullptr);
      node.val1 = notcond.res();

      rewriter.setInsertionPoint(&state1.getBody()->back());
      auto notcond1 = rewriter.create<hec::NotOp>(
          state1.getLoc(), rewriter.getI1Type(), mapOp2Reg(forop), nullptr);
      node.val3 = notcond1.res();

      assert(node.edges.size() == 2 && "FOR Node must have two successors");
      auto for_end = node.edges[1].to;

      auto do_end = gen_states(node.edges[0].to, for_end, reach);

      node.tthen = do_end;
      std::cerr << "for_end is " << do_end
                << ", state id = " << node2State[do_end] << std::endl;

      ret = gen_states(for_end, tend, reach);

      auto clock = getClock(node.edges[0]);

      std::vector<hec::StateOp> bags0;
      std::vector<hec::StateOp> bags1;
      bags0.push_back(state0);
      bags1.push_back(state1);

      assert(clock >= 1);

      for (unsigned long i = 0; i < clock - 1; i++) {
        auto curState = bags0[i];
        auto nextState = createStateAfter(
            std::to_string(id) + std::string("_") + std::to_string(i),
            curState);
        bags0.push_back(nextState);
        if (i)
          createGoto(curState,
                     std::to_string(id) + std::string("_") + std::to_string(i));
        else
          createGotoCond(
              curState,
              std::to_string(id) + std::string("_") + std::to_string(i), cond);
        curState = bags1[i];
        nextState = createStateAfter(
            std::to_string(id) + std::string("_entry_") + std::to_string(i),
            curState);
        bags1.push_back(nextState);

        if (i)
          createGoto(curState, std::to_string(id) + std::string("_entry_") +
                                   std::to_string(i));
        else
          createGotoCond(curState,
                         std::to_string(id) + std::string("_entry_") +
                             std::to_string(i),
                         mapOp2Reg(forop));
      }

      unsigned long nextNodeId;
      nextNodeId = node.edges[0].to;
      std::string nextname = std::to_string(node2State[nextNodeId]);
      createGoto(bags0.back(), nextname);
      createGoto(bags1.back(), nextname);

      createGotoCond(state0, std::to_string(node2State[for_end]), notcond);
      createGotoCond(state1, std::to_string(node2State[for_end]), notcond1);

      createGoto(str2State[std::to_string(node2State[do_end])],
                 std::to_string(id) + std::string("_entry"));
      set_state_for(t, state0.getName().str(), forop);
    }

    // std::cerr << "node " << node.t << " return " << ret << std::endl;
    return ret;
  }

  auto gen_constants() {
    for (auto &vud : valueUseDefs)
      if (vud.type == ValueUseDef::Type::Constant) {
        auto oldconst = llvm::dyn_cast<mlir::ConstantOp>(vud.owner);

        auto oldattr = oldconst.getValue();

        rewriter.setInsertionPoint(&component.getBody()->back());
        auto constop =
            rewriter.create<mlir::ConstantOp>(component.getLoc(), oldattr);

        vud.owner = constop;
        constop.dump();
      }
  }

  auto gen_argument_backup() {
    for (auto vud : valueUseDefs)
      if (vud.type == ValueUseDef::Type::Argument) {
        mlir::Value src;
        for (auto pr : arg2arg)
          if (pr.first == vud.value)
            src = pr.second;

        std::cerr << "vud.id = " << vud.id
                  << " , registerId = " << value2Reg[vud.id] << ": ";

        registers[value2Reg[vud.id]].op.dump();

        auto state0 = str2State[std::string("0")];
        rewriter.setInsertionPoint(&state0.getBody()->back());
        rewriter.create<hec::AssignOp>(
            state0.getLoc(), registers[value2Reg[vud.id]].op.getResult(0), src,
            nullptr);
      }
  }

  auto createNot(mlir::Value &cond0, mlir::Value cond, hec::StateOp state0) {
    bool found = 0;
    for (auto notop : state0.getBody()->getOps<hec::NotOp>())
      if (notop.src() == cond) {
        cond0 = notop.res();
        found = 1;
      }
    if (!found) {
      rewriter.setInsertionPoint(&state0.getBody()->back());
      auto notop = rewriter.create<hec::NotOp>(
          state0.getLoc(), rewriter.getI1Type(), cond, nullptr);
      cond0 = notop.res();
    }
  }

  auto insertLoadOp(hec::StateOp state0, hec::StateOp state1, tor::LoadOp load,
                    mlir::Value cond, bool needNot = 0,
                    mlir::Value cond2 = nullptr) {
    std::cerr << "Insert LoadOp on " << state0.getName().str() << std::endl;
    load.dump();

    mlir::Value cond0(cond), cond1(cond);
    if (needNot) {
      createNot(cond0, cond, state0);
      createNot(cond1, cond, state1);
    }

    if (cond2 != nullptr)
      cond1 = cond2;

    auto t = load.starttime();
    auto mem = mapValueMem(load.memref());

    assert(mem.id != -1ul);

    auto indices = load.indices();
    assert(indices.size() == 1 && "Require 1 indice for LoadOp");

    auto address = mem.getAddress();
    auto r_en = mem.getReadEnable();
    auto r_data = mem.getReadData();

    auto indice = mapValue(indices.front(), t, state0.getName().str());

    auto res = mapValue(load.getResult(), load.endtime());
    assert(indice != nullptr && address != nullptr && r_en != nullptr &&
           r_data != nullptr);
    assert(res != nullptr);

    auto &vud = getVUD(load.getResult(), load);
    vud.wire = r_data;

    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<mlir::hec::AssignOp>(state0.getLoc(), address, indice,
                                         cond0);
    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<mlir::hec::EnableOp>(state0.getLoc(), r_en, cond0);
    rewriter.setInsertionPoint(&state1.getBody()->back());
    rewriter.create<mlir::hec::AssignOp>(state1.getLoc(), res, r_data, cond1);
  }

  auto insertStoreOp(hec::StateOp state0, hec::StateOp state1,
                     tor::StoreOp store, mlir::Value cond, bool needNot = 0,
                     mlir::Value cond2 = nullptr) {
    store.dump();

    mlir::Value cond0(cond), cond1(cond);
    if (needNot) {
      createNot(cond0, cond, state0);
      createNot(cond1, cond, state1);
    }

    if (cond2 != nullptr)
      cond1 = cond2;

    auto t = store.starttime();
    auto mem = mapValueMem(store.memref());
    auto indices = store.indices();
    assert(indices.size() == 1 && "Require 1 indice for StoreOp");

    auto address = mem.getAddress();
    auto w_en = mem.getWriteEnable();
    auto w_data = mem.getWriteData();

    auto indice = mapValue(indices.front(), t, state0.getName().str());

    auto operand =
        mapValue(store.value(), store.starttime(), state0.getName().str());
    assert(indice != nullptr && address != nullptr && w_en != nullptr &&
           w_data != nullptr);
    assert(operand != nullptr);

    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<mlir::hec::AssignOp>(state0.getLoc(), address, indice,
                                         cond0);
    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<mlir::hec::AssignOp>(state0.getLoc(), w_data, operand,
                                         cond0);
    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<mlir::hec::EnableOp>(state0.getLoc(), w_en, cond0);
  }

  auto insertCmpIOp(hec::StateOp state0, tor::CmpIOp bop,
                    mlir::Value cond = nullptr, bool needNot = 0,
                    std::string str = "") {
    bop.dump();
    mlir::Value cond0(cond);
    if (needNot)
      createNot(cond0, cond, state0);

    auto t = bop.starttime();
    auto lhs = mapValue(bop.lhs(), t, state0.getName().str());
    auto rhs = mapValue(bop.rhs(), t, state0.getName().str());

    assert(lhs != nullptr && rhs != nullptr);

    auto res = mapValue(bop.getResult(), bop.endtime());

    rewriter.setInsertionPoint(&state0.getBody()->back());
    auto newOp = rewriter.create<hec::CmpIOp>(
        state0.getLoc(), bop.getResult().getType(), lhs, rhs,
        rewriter.getStringAttr(str), cond0);

    auto &vud = getVUD(bop.getResult(), bop);
    vud.wire = newOp.getResult();

    std::cerr << "feed into vud.wire" << std::endl;

    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<hec::AssignOp>(state0.getLoc(), res, newOp.getResult(),
                                   cond0);
  }

  template <typename OldType, typename NewType>
  auto insertCombUnaryOp(hec::StateOp state0, OldType bop,
                         mlir::Value cond = nullptr, bool needNot = 0,
                         std::string str = "") {
    bop.dump();
    mlir::Value cond0(cond);
    if (needNot)
      createNot(cond0, cond, state0);

    auto t = bop->template getAttrOfType<IntegerAttr>("starttime").getInt();

    assert(state0 != nullptr);
    auto lhs = mapValue(bop->getOperand(0), t, state0.getName().str());

    assert(lhs != nullptr);

    auto res =
        mapValue(bop.getResult(),
                 bop->template getAttrOfType<IntegerAttr>("endtime").getInt());

    rewriter.setInsertionPoint(&state0.getBody()->back());

    auto newOp = rewriter.create<NewType>(
        state0.getLoc(), bop.getResult().getType(), lhs, cond0);

    auto &vud = getVUD(bop.getResult(), bop);
    vud.wire = newOp.getResult();

    std::cerr << "feed into vud.wire" << std::endl;

    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<hec::AssignOp>(state0.getLoc(), res, newOp.getResult(),
                                   cond0);
  }

  template <typename OldType, typename NewType>
  auto insertCombBinaryOp(hec::StateOp state0, OldType bop,
                          mlir::Value cond = nullptr, bool needNot = 0,
                          std::string str = "") {
    bop.dump();
    mlir::Value cond0(cond);
    if (needNot)
      createNot(cond0, cond, state0);

    auto t = bop->template getAttrOfType<IntegerAttr>("starttime").getInt();
    // auto t = bop.starttime();

    assert(state0 != nullptr);
    auto lhs = mapValue(bop.lhs(), t, state0.getName().str());
    auto rhs = mapValue(bop.rhs(), t, state0.getName().str());

    assert(lhs != nullptr && rhs != nullptr);

    auto res =
        mapValue(bop.getResult(),
                 bop->template getAttrOfType<IntegerAttr>("endtime").getInt());

    rewriter.setInsertionPoint(&state0.getBody()->back());

    auto newOp = rewriter.create<NewType>(
        state0.getLoc(), bop.getResult().getType(), lhs, rhs, cond0);

    auto &vud = getVUD(bop.getResult(), bop);
    vud.wire = newOp.getResult();

    std::cerr << "feed into vud.wire" << std::endl;

    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<hec::AssignOp>(state0.getLoc(), res, newOp.getResult(),
                                   cond0);
  }

  auto insertSelectOp(hec::StateOp state0, SelectOp bop,
                      mlir::Value cond = nullptr, bool needNot = 0,
                      std::string str = "") {
    bop.dump();
    mlir::Value cond0(cond);
    if (needNot)
      createNot(cond0, cond, state0);

    auto t = bop->template getAttrOfType<IntegerAttr>("starttime").getInt();
    // auto t = bop.starttime();

    assert(state0 != nullptr);
    auto condition = mapValue(bop.condition(), t, state0.getName().str());
    auto lhs = mapValue(bop.true_value(), t, state0.getName().str());
    auto rhs = mapValue(bop.false_value(), t, state0.getName().str());

    assert(condition != nullptr && lhs != nullptr && rhs != nullptr);

    auto res =
        mapValue(bop.getResult(),
                 bop->template getAttrOfType<IntegerAttr>("endtime").getInt());

    rewriter.setInsertionPoint(&state0.getBody()->back());

    auto newOp = rewriter.create<hec::SelectOp>(
        state0.getLoc(), bop.getResult().getType(), condition, lhs, rhs, cond0);

    auto &vud = getVUD(bop.getResult(), bop);
    vud.wire = newOp.getResult();

    std::cerr << "feed into vud.wire" << std::endl;

    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<hec::AssignOp>(state0.getLoc(), res, newOp.getResult(),
                                   cond0);
  }

  template <typename OpType>
  auto insertMultiCycleUnaryOp(hec::StateOp state0, hec::StateOp state1,
                               OpType bop, mlir::Value cond, bool needNot = 0,
                               mlir::Value cond2 = nullptr) {
    bop.dump();

    mlir::Value cond0(cond), cond1(cond);

    if (needNot) {
      createNot(cond0, cond, state0);
      createNot(cond1, cond, state1);
    }

    if (cond2 != nullptr)
      cond1 = cond2;

    auto t = bop->template getAttrOfType<IntegerAttr>("starttime").getInt();
    auto lhs = mapValue(bop.in(), t, state0.getName().str());

    assert(lhs != nullptr);

    auto res =
        mapValue(bop.getResult(),
                 bop->template getAttrOfType<IntegerAttr>("endtime").getInt());
    res.dump();

    auto primitive = getCellByOp(bop).primitive;
    primitive.dump();

    auto &vud = getVUD(bop.getResult(), bop);
    vud.wire = primitive.getResult(1);

    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<mlir::hec::AssignOp>(state0.getLoc(),
                                         primitive.getResult(0), lhs, cond0);
    rewriter.setInsertionPoint(&state1.getBody()->back());
    rewriter.create<mlir::hec::AssignOp>(state1.getLoc(), res,
                                         primitive.getResult(1), cond1);
  }

  template <typename OpType>
  auto insertMultiCycleBinaryOp(hec::StateOp state0, hec::StateOp state1,
                                OpType bop, mlir::Value cond, bool needNot = 0,
                                mlir::Value cond2 = nullptr) {
    bop.dump();

    mlir::Value cond0(cond), cond1(cond);

    if (needNot) {
      createNot(cond0, cond, state0);
      createNot(cond1, cond, state1);
    }

    if (cond2 != nullptr)
      cond1 = cond2;

    // auto t = bop.starttime();
    auto t = bop->template getAttrOfType<IntegerAttr>("starttime").getInt();
    auto lhs = mapValue(bop.lhs(), t, state0.getName().str());
    auto rhs = mapValue(bop.rhs(), t, state0.getName().str());

    assert(lhs != nullptr);
    assert(rhs != nullptr);

    auto res =
        mapValue(bop.getResult(),
                 bop->template getAttrOfType<IntegerAttr>("endtime").getInt());
    res.dump();

    auto primitive = getCellByOp(bop).primitive;
    primitive.dump();

    auto &vud = getVUD(bop.getResult(), bop);
    vud.wire = primitive.getResult(2);

    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<mlir::hec::AssignOp>(state0.getLoc(),
                                         primitive.getResult(0), lhs, cond0);
    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<mlir::hec::AssignOp>(state0.getLoc(),
                                         primitive.getResult(1), rhs, cond0);
    rewriter.setInsertionPoint(&state1.getBody()->back());
    rewriter.create<mlir::hec::AssignOp>(state1.getLoc(), res,
                                         primitive.getResult(2), cond1);
  }

  auto insertSeqOp(mlir::Operation *op, size_t from, size_t to) {
    auto state0 = str2State[std::to_string(node2State[from])];
    assert(nodes[to].backEdges.size() == 1 && "Single backedge for Seq");

    // std::string toname =
    //     std::to_string(node2State[nodes[to].backEdges[0].from]);
    // auto clock = getClock(nodes[to].backEdges[0]);
    // assert(clock >= 1 && "Clock >= 1 for Seq");
    // if (clock > 1)
    //   toname += std::string("_") + std::to_string(clock - 2);

    std::string toname = std::to_string(node2State[to]);

    auto state1 = str2State[toname];

    // std::cerr << "insert op, s" << std::to_string(from) << " -> s" << toname
    //           << ": ";

#define BINDCOM(OpType, NewType)                                               \
  if (auto sop = llvm::dyn_cast<OpType>(op))                                   \
    insertCombBinaryOp<OpType, NewType>(state0, sop, nullptr);

    BINDCOM(tor::AddIOp, hec::AddIOp)
    BINDCOM(tor::SubIOp, hec::SubIOp)
    BINDCOM(AndOp, hec::AndOp)
    BINDCOM(OrOp, hec::OrOp)
    BINDCOM(XOrOp, hec::XOrOp)
    BINDCOM(ShiftLeftOp, hec::ShiftLeftOp)
    BINDCOM(SignedShiftRightOp, hec::SignedShiftRightOp)
#undef BINDCOM

#define BINDCOMU(OpType, NewType)                                              \
  if (auto sop = llvm::dyn_cast<OpType>(op))                                   \
    insertCombUnaryOp<OpType, NewType>(state0, sop, nullptr);

    BINDCOMU(NegFOp, hec::NegFOp)
    BINDCOMU(TruncateIOp, hec::TruncateIOp)
    BINDCOMU(SignExtendIOp, hec::SignExtendIOp)
#undef BINDCOMU

    if (auto sop = llvm::dyn_cast<SelectOp>(op))
      insertSelectOp(state0, sop, nullptr);

#define BINDMULTICYCLE(OpType)                                                 \
  if (auto sop = llvm::dyn_cast<OpType>(op))                                   \
    insertMultiCycleBinaryOp(state0, state1, sop, nullptr);

    BINDMULTICYCLE(tor::MulIOp)
    BINDMULTICYCLE(tor::AddFOp)
    BINDMULTICYCLE(tor::SubFOp)
    BINDMULTICYCLE(tor::MulFOp)
    BINDMULTICYCLE(tor::DivFOp)
    BINDMULTICYCLE(SignedDivIOp)

#undef BINDMULTICYCLE

#define BINDMULTICYCLEU(OpType)                                                \
  if (auto sop = llvm::dyn_cast<OpType>(op))                                   \
    insertMultiCycleUnaryOp<OpType>(state0, state1, sop, nullptr);
    BINDMULTICYCLEU(SIToFPOp)
    BINDMULTICYCLEU(FPToSIOp)
#undef BINDMULTICYCLEU

    if (auto cmpf = llvm::dyn_cast<tor::CmpFOp>(op))
      insertMultiCycleBinaryOp(state0, state1, cmpf, nullptr);
    if (auto cmpi = llvm::dyn_cast<tor::CmpIOp>(op))
      insertCmpIOp(state0, cmpi, nullptr, 0,
                   tor::stringifyEnum(cmpi.predicate()).str());

    if (auto load = llvm::dyn_cast<tor::LoadOp>(op))
      insertLoadOp(state0, state1, load, nullptr);
    if (auto store = llvm::dyn_cast<tor::StoreOp>(op))
      insertStoreOp(state0, state1, store, nullptr);
    std::cerr << std::endl;
  }

  auto insertCallOp(size_t t) {
    auto node = nodes.at(t);
    assert(node.opsOnEdge.size() == 1 && "One opOnEdge for CALL");

    auto state0 = str2State[std::to_string(node2State[t])];
    auto state1 = str2State[std::to_string(node2State[t]) + "_wait"];

    auto instance = callDict[t];
    auto callop = llvm::dyn_cast<tor::CallOp>(node.op);

    auto ptr = instance.getResults().begin();
    for (auto operand : callop.getOperands()) {
      rewriter.setInsertionPoint(&state0.getBody()->back());
      rewriter.create<hec::AssignOp>(
          state0.getLoc(), *ptr++, mapValue(operand, t, state0.getName().str()),
          nullptr);
    }

    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<hec::GoOp>(state0.getLoc(), instance.instanceName(),
                               nullptr);
    ptr++;

    for (auto result : callop.getResults())
      if (mapValue(result, t) != nullptr) {
        rewriter.setInsertionPoint(&state0.getBody()->back());
        rewriter.create<hec::AssignOp>(state0.getLoc(), mapValue(result, t),
                                       *ptr, instance.getResults().back());
        rewriter.setInsertionPoint(&state1.getBody()->back());
        rewriter.create<hec::AssignOp>(state1.getLoc(), mapValue(result, t),
                                       *ptr++, instance.getResults().back());
      }
  }

  auto insertIfOp(size_t t) {
    auto &node = nodes.at(t);
    auto ifop = llvm::dyn_cast<tor::IfOp>(node.op);

    // std::cerr << "Insert IfOp: " << t << " -> " << node.tthen << ","
    //           << node.telse << " -> " << node.tend << std::endl;
    auto &state0 = str2State[std::to_string(node2State[t])];
    // auto &state1 = str2State[std::to_string(node2State[node.tend])];
    // ThenGen
    if (node.tthen != node.tend) {
      auto nextnode = node.edges[0].to;
      // std::cerr << "then : " << nextnode << " -> " << node.tthen <<
      // std::endl;

      for (auto opoe : node.opsOnEdge)
        if (opoe.to == nextnode) {
          auto to = opoe.to;
          auto toname = std::to_string(node.t);
          auto clock = getClock(nodes[to].backEdges[0]);
          assert(clock >= 1 && "Clock >= 1 for Seq");
          if (clock > 1)
            toname += std::string("_then_") + std::to_string(clock - 2);
          auto &state_then = str2State[toname];

#define BINDCOMBTHEN(OpType, NewType)                                          \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op))                              \
    insertCombBinaryOp<OpType, NewType>(state0, sop,                           \
                                        mapValue(ifop.condition(), t));
          BINDCOMBTHEN(tor::AddIOp, hec::AddIOp)
          BINDCOMBTHEN(tor::SubIOp, hec::SubIOp)
          BINDCOMBTHEN(AndOp, hec::AndOp)
          BINDCOMBTHEN(OrOp, hec::OrOp)
          BINDCOMBTHEN(XOrOp, hec::XOrOp)
          BINDCOMBTHEN(ShiftLeftOp, hec::ShiftLeftOp)
          BINDCOMBTHEN(SignedShiftRightOp, hec::SignedShiftRightOp)

#undef BINDCOMBTHEN

#define BINDCOMBUTHEN(OpType, NewType)                                         \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op))                              \
    insertCombUnaryOp<OpType, NewType>(state0, sop,                            \
                                       mapValue(ifop.condition(), t));
          BINDCOMBUTHEN(NegFOp, hec::NegFOp)
          BINDCOMBUTHEN(TruncateIOp, hec::TruncateIOp)
          BINDCOMBUTHEN(SignExtendIOp, hec::SignExtendIOp)
#undef BINDCOMBUTHEN

          if (auto sop = llvm::dyn_cast<SelectOp>(opoe.op))
            insertSelectOp(state0, sop, mapValue(ifop.condition(), t));

#define BINDMULTICYCLEUTHEN(OpType)                                            \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op))                              \
    insertMultiCycleUnaryOp(state0, state_then, sop,                           \
                            mapValue(ifop.condition(), t));
          BINDMULTICYCLEUTHEN(SIToFPOp)
          BINDMULTICYCLEUTHEN(FPToSIOp)
#undef BINDMULTICYCLEUTHEN

#define BINDMULTICYCLETHEN(OpType)                                             \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op))                              \
    insertMultiCycleBinaryOp(state0, state_then, sop,                          \
                             mapValue(ifop.condition(), t));
          BINDMULTICYCLETHEN(tor::MulIOp)
          BINDMULTICYCLETHEN(tor::AddFOp)
          BINDMULTICYCLETHEN(tor::SubFOp)
          BINDMULTICYCLETHEN(tor::MulFOp)
          BINDMULTICYCLETHEN(tor::DivFOp)
          BINDMULTICYCLETHEN(SignedDivIOp)
#undef BINDMULTICYCLETHEN

          if (auto cmpf = llvm::dyn_cast<tor::CmpFOp>(opoe.op))
            insertMultiCycleBinaryOp(state0, state_then, cmpf,
                                     mapValue(ifop.condition(), t));
          if (auto cmpi = llvm::dyn_cast<tor::CmpIOp>(opoe.op))
            insertCmpIOp(state0, cmpi, mapValue(ifop.condition(), t), 0,
                         tor::stringifyEnum(cmpi.predicate()).str());

          if (auto load = llvm::dyn_cast<tor::LoadOp>(opoe.op))
            insertLoadOp(state0, state_then, load,
                         mapValue(ifop.condition(), t));
          if (auto store = llvm::dyn_cast<tor::StoreOp>(opoe.op))
            insertStoreOp(state0, state_then, store,
                          mapValue(ifop.condition(), t));
        }

      std::cerr << "yield then at node " << node.tthen << std::endl;
      auto &state_then_end = str2State[std::to_string(node2State[node.tthen])];
      auto yield = llvm::dyn_cast<tor::YieldOp>(
          ifop.thenRegion().back().getTerminator());

      auto ptr = ifop.getResults().begin();
      for (auto operand : yield.getOperands()) {
        rewriter.setInsertionPoint(&state_then_end.getBody()->back());
        rewriter.create<hec::AssignOp>(state_then_end.getLoc(),
                                       mapValue(*ptr, node.tthen),
                                       mapValue(operand, node.tthen), nullptr);
      }
    } else {
      auto yield = llvm::dyn_cast<tor::YieldOp>(
          ifop.thenRegion().back().getTerminator());

      auto ptr = ifop.getResults().begin();
      for (auto operand : yield.getOperands()) {
        rewriter.setInsertionPoint(&state0.getBody()->back());
        rewriter.create<hec::AssignOp>(state0.getLoc(), mapValue(*ptr, node.t),
                                       mapValue(operand, node.t),
                                       mapValue(ifop.condition(), t));
      }
    }

    // ElseGen
    if (node.edges.size() == 2) {
      if (node.telse != node.tend) {
        auto nextnode = node.edges[1].to;
        // std::cerr << "else : " << nextnode << " -> " << node.telse <<
        // std::endl;

        for (auto opoe : node.opsOnEdge)
          if (opoe.to == nextnode) {
            auto to = opoe.to;
            auto toname = std::to_string(node.t);
            auto clock = getClock(nodes[to].backEdges[0]);
            assert(clock >= 1 && "Clock >= 1 for Seq");
            if (clock > 1)
              toname += std::string("_else_") + std::to_string(clock - 2);
            auto &state_else = str2State[toname];

#define BINDCOMBELSE(OpType, NewType)                                          \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op))                              \
    insertCombBinaryOp<OpType, NewType>(state0, sop,                           \
                                        mapValue(ifop.condition(), t), 1);
            BINDCOMBELSE(tor::AddIOp, hec::AddIOp)
            BINDCOMBELSE(tor::SubIOp, hec::SubIOp)
            BINDCOMBELSE(AndOp, hec::AndOp)
            BINDCOMBELSE(OrOp, hec::OrOp)
            BINDCOMBELSE(XOrOp, hec::XOrOp)
            BINDCOMBELSE(ShiftLeftOp, hec::ShiftLeftOp)
            BINDCOMBELSE(SignedShiftRightOp, hec::SignedShiftRightOp)

#undef BINDCOMBELSE

#define BINDCOMBUELSE(OpType, NewType)                                         \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op))                              \
    insertCombUnaryOp<OpType, NewType>(state0, sop,                            \
                                       mapValue(ifop.condition(), t), 1);
            BINDCOMBUELSE(NegFOp, hec::NegFOp)
            BINDCOMBUELSE(TruncateIOp, hec::TruncateIOp)
            BINDCOMBUELSE(SignExtendIOp, hec::SignExtendIOp)
#undef BINDCOMBUELSE

            if (auto sop = llvm::dyn_cast<SelectOp>(opoe.op))
              insertSelectOp(state0, sop, mapValue(ifop.condition(), t), 1);

#define BINDMULTICYCLEUELSE(OpType)                                            \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op))                              \
    insertMultiCycleUnaryOp(state0, state_else, sop,                           \
                            mapValue(ifop.condition(), t), 1);
            BINDMULTICYCLEUELSE(SIToFPOp)
            BINDMULTICYCLEUELSE(FPToSIOp)
#undef BINDMULTICYCLEUELSE

#define BINDMULTICYCLEELSE(OpType)                                             \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op))                              \
    insertMultiCycleBinaryOp(state0, state_else, sop,                          \
                             mapValue(ifop.condition(), t), 1);
            BINDMULTICYCLEELSE(tor::MulIOp)
            BINDMULTICYCLEELSE(tor::AddFOp)
            BINDMULTICYCLEELSE(tor::SubFOp)
            BINDMULTICYCLEELSE(tor::MulFOp)
            BINDMULTICYCLEELSE(tor::DivFOp)
            BINDMULTICYCLEELSE(SignedDivIOp)
#undef BINDMULTICYCLEELSE

            if (auto cmpf = llvm::dyn_cast<tor::CmpFOp>(opoe.op))
              insertMultiCycleBinaryOp(state0, state_else, cmpf,
                                       mapValue(ifop.condition(), t), 1);
            if (auto cmpi = llvm::dyn_cast<tor::CmpIOp>(opoe.op))
              insertCmpIOp(state0, cmpi, mapValue(ifop.condition(), t), 1,
                           tor::stringifyEnum(cmpi.predicate()).str());

            if (auto load = llvm::dyn_cast<tor::LoadOp>(opoe.op))
              insertLoadOp(state0, state_else, load,
                           mapValue(ifop.condition(), t), 1);
            if (auto store = llvm::dyn_cast<tor::StoreOp>(opoe.op))
              insertStoreOp(state0, state_else, store,
                            mapValue(ifop.condition(), t), 1);
          }

        auto &state_else_end =
            str2State[std::to_string(node2State[node.telse])];
        auto yield = llvm::dyn_cast<tor::YieldOp>(
            ifop.elseRegion().back().getTerminator());

        auto ptr = ifop.getResults().begin();
        for (auto operand : yield.getOperands()) {
          rewriter.setInsertionPoint(&state_else_end.getBody()->back());
          rewriter.create<hec::AssignOp>(
              state_else_end.getLoc(), mapValue(*ptr, node.telse),
              mapValue(operand, node.telse), nullptr);
        }
      } else {
        auto yield = llvm::dyn_cast<tor::YieldOp>(
            ifop.elseRegion().back().getTerminator());

        auto ptr = ifop.getResults().begin();
        for (auto operand : yield.getOperands()) {
          rewriter.setInsertionPoint(&state0.getBody()->back());
          rewriter.create<hec::AssignOp>(state0.getLoc(),
                                         mapValue(*ptr, node.t),
                                         mapValue(operand, node.t), node.val);
        }
      }
    }
  }

  auto insertWhileOp(size_t t) {
    auto &node = nodes.at(t);
    auto whileop = llvm::dyn_cast<tor::WhileOp>(node.op);

    auto &state0 = str2State[std::to_string(node2State[t])];
    auto &state_entry = str2State[std::to_string(node2State[t]) + "_entry"];
    auto &state_do_end = str2State[std::to_string(node.tthen)];

    // First enter
    std::vector<std::pair<mlir::Value, mlir::Value>> val2val;

    auto ptr = whileop.getOperands().begin();
    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<hec::AssignOp>(
        state0.getLoc(), mapValue(whileop.getRegion(0).getArgument(0), t),
        mapValue(*ptr++, t), nullptr);

    for (auto operand : whileop.getRegion(1).getArguments()) {
      rewriter.setInsertionPoint(&state0.getBody()->back());
      val2val.push_back(
          std::make_pair(mapValue(*ptr, t), mapValue(operand, t)));
      rewriter.create<hec::AssignOp>(state0.getLoc(), mapValue(operand, t),
                                     mapValue(*ptr++, t), nullptr);
    }

    // Set ops

    for (auto opoe : node.opsOnEdge) {
      auto to = opoe.to;
      auto clock = getClock(nodes[to].backEdges[0]);
      auto toname0 = std::to_string(node.t);
      auto toname1 = std::to_string(node.t) + "_entry";
      assert(clock >= 1 && "Clock >= 1 for Seq");
      if (clock > 1) {
        toname0 += std::string("_") + std::to_string(clock - 2);
        toname1 += std::string("_") + std::to_string(clock - 2);
      }
      auto &state0_next = str2State[toname0];
      auto &state_entry_next = str2State[toname1];

#define BINDCOMBDO(OpType, NewType)                                            \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op)) {                            \
    insertCombBinaryOp<OpType, NewType>(state0, sop,                           \
                                        mapValue(whileop.getOperand(0), t));   \
    insertCombBinaryOp<OpType, NewType>(                                       \
        state_entry, sop, mapValue(whileop.getRegion(0).getArgument(0), t));   \
  }
      BINDCOMBDO(tor::AddIOp, hec::AddIOp)
      BINDCOMBDO(tor::SubIOp, hec::SubIOp)
      BINDCOMBDO(AndOp, hec::AndOp)
      BINDCOMBDO(OrOp, hec::OrOp)
      BINDCOMBDO(XOrOp, hec::XOrOp)
      BINDCOMBDO(ShiftLeftOp, hec::ShiftLeftOp)
      BINDCOMBDO(SignedShiftRightOp, hec::SignedShiftRightOp)
#undef BINDCOMBDO

#define BINDCOMBUDO(OpType, NewType)                                           \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op)) {                            \
    insertCombUnaryOp<OpType, NewType>(state0, sop,                            \
                                       mapValue(whileop.getOperand(0), t));    \
    insertCombUnaryOp<OpType, NewType>(                                        \
        state_entry, sop, mapValue(whileop.getRegion(0).getArgument(0), t));   \
  }
      BINDCOMBUDO(NegFOp, hec::NegFOp)
      BINDCOMBUDO(TruncateIOp, hec::TruncateIOp)
      BINDCOMBUDO(SignExtendIOp, hec::SignExtendIOp)
#undef BINDCOMBUDO

      if (auto sop = llvm::dyn_cast<SelectOp>(opoe.op)) {
        insertSelectOp(state0, sop, mapValue(whileop.getOperand(0), t));
        insertSelectOp(state_entry, sop,
                       mapValue(whileop.getRegion(0).getArgument(0), t));
      }

#define BINDMULTICYCLEUDO(OpType)                                              \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op)) {                            \
    insertMultiCycleUnaryOp(state0, state0_next, sop,                          \
                            mapValue(whileop.getOperand(0), t));               \
    insertMultiCycleUnaryOp(state_entry, state_entry_next, sop,                \
                            mapValue(whileop.getRegion(0).getArgument(0), t)); \
  }
      BINDMULTICYCLEUDO(SIToFPOp)
      BINDMULTICYCLEUDO(FPToSIOp)

#undef BINDMULTICYCLEUDO

#define BINDMULTICYCLEDO(OpType)                                               \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op)) {                            \
    insertMultiCycleBinaryOp(state0, state0_next, sop,                         \
                             mapValue(whileop.getOperand(0), t));              \
    insertMultiCycleBinaryOp(                                                  \
        state_entry, state_entry_next, sop,                                    \
        mapValue(whileop.getRegion(0).getArgument(0), t));                     \
  }
      BINDMULTICYCLEDO(tor::MulIOp)
      BINDMULTICYCLEDO(tor::AddFOp)
      BINDMULTICYCLEDO(tor::SubFOp)
      BINDMULTICYCLEDO(tor::MulFOp)
      BINDMULTICYCLEDO(tor::DivFOp)
      BINDMULTICYCLEDO(SignedDivIOp)

#undef BINDMULTICYCLEDO

      if (auto cmpf = llvm::dyn_cast<tor::CmpFOp>(opoe.op)) {
        insertMultiCycleBinaryOp(state0, state0_next, cmpf,
                                 mapValue(whileop.getOperand(0), t));
        insertMultiCycleBinaryOp(
            state_entry, state_entry_next, cmpf,
            mapValue(whileop.getRegion(0).getArgument(0), t));
      }

      if (auto cmpi = llvm::dyn_cast<tor::CmpIOp>(opoe.op)) {
        insertCmpIOp(state0, cmpi, mapValue(whileop.getOperand(0), t), 0,
                     tor::stringifyEnum(cmpi.predicate()).str());
        insertCmpIOp(state_entry, cmpi,
                     mapValue(whileop.getRegion(0).getArgument(0), t), 0,
                     tor::stringifyEnum(cmpi.predicate()).str());
      }
      if (auto load = llvm::dyn_cast<tor::LoadOp>(opoe.op)) {
        insertLoadOp(state0, state0_next, load,
                     mapValue(whileop.getOperand(0), t));
        insertLoadOp(state_entry, state_entry_next, load,
                     mapValue(whileop.getRegion(0).getArgument(0), t));
      }
      if (auto store = llvm::dyn_cast<tor::StoreOp>(opoe.op)) {
        insertStoreOp(state0, state0_next, store,
                      mapValue(whileop.getOperand(0), t));
        insertStoreOp(state_entry, state_entry_next, store,
                      mapValue(whileop.getRegion(0).getArgument(0), t));
      }
    }

    // Set yield
    auto yield = llvm::dyn_cast<tor::YieldOp>(
        whileop.getRegion(1).back().getTerminator());
    auto ptrItr = whileop.getRegion(1).getArguments().begin();

    yield.dump();
    ptrItr->dump();

    bool isCond = 0;
    for (auto operand : yield.getOperands()) {
      // operand.dump();
      if (!isCond) {
        isCond = 1;
        rewriter.setInsertionPoint(&state_do_end.getBody()->back());
        rewriter.create<hec::AssignOp>(
            state_do_end.getLoc(),
            mapValue(whileop.getRegion(0).getArgument(0), node.tthen),
            mapValue(operand, node.tthen), nullptr);
        continue;
      }
      rewriter.setInsertionPoint(&state_do_end.getBody()->back());
      rewriter.create<hec::AssignOp>(state_do_end.getLoc(),
                                     mapValue(*ptrItr++, node.tthen),
                                     mapValue(operand, node.tthen), nullptr);
    }

    // Set exit
    ptrItr = whileop.getRegion(1).getArguments().begin();
    for (auto result : whileop.getResults()) {
      rewriter.setInsertionPoint(&state_entry.getBody()->back());
      rewriter.create<hec::AssignOp>(state_entry.getLoc(), mapValue(result, t),
                                     mapValue(*ptrItr++, t), node.val1);
    }
    ptrItr = whileop.getRegion(1).getArguments().begin();
    for (auto result : whileop.getResults()) {
      rewriter.setInsertionPoint(&state0.getBody()->back());
      rewriter.create<hec::AssignOp>(state0.getLoc(), mapValue(result, t),
                                     mapValue(*ptrItr++, t), node.val0);
    }
  }

  auto insertForOp(size_t t) {
    auto &node = nodes.at(t);
    auto forop = llvm::dyn_cast<tor::ForOp>(node.op);
    auto &state0 = str2State[std::to_string(node2State[t])];
    auto &state_entry = str2State[std::to_string(node2State[t]) + "_entry"];
    auto &state_do_end = str2State[std::to_string(node.tthen)];

    // First enter
    std::vector<std::pair<mlir::Value, mlir::Value>> val2val;

    rewriter.setInsertionPoint(&state0.getBody()->back());
    rewriter.create<hec::AssignOp>(state0.getLoc(),
                                   mapValue(forop.getInductionVar(), t),
                                   mapValue(forop.lowerBound(), t), nullptr);

    auto ptr = forop.getIterOperands().begin();
    for (auto operand : forop.getRegionIterArgs()) {
      rewriter.setInsertionPoint(&state0.getBody()->back());
      rewriter.create<hec::AssignOp>(state0.getLoc(), mapValue(operand, t),
                                     mapValue(*ptr++, t), nullptr);
    }

    // Set ops
    for (auto opoe : node.opsOnEdge) {
      auto to = opoe.to;
      auto clock = getClock(nodes[to].backEdges[0]);
      auto toname0 = std::to_string(node.t);
      auto toname1 = std::to_string(node.t) + "_entry";
      assert(clock >= 1 && "Clock >= 1 for Seq");
      if (clock > 1) {
        toname0 += std::string("_") + std::to_string(clock - 2);
        toname1 += std::string("_") + std::to_string(clock - 2);
      }
      auto &state0_next = str2State[toname0];
      auto &state_entry_next = str2State[toname1];
#define BINDCOMBDO(OpType, NewType)                                            \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op)) {                            \
    insertCombBinaryOp<OpType, NewType>(state0, sop, node.val0);               \
    insertCombBinaryOp<OpType, NewType>(state_entry, sop, mapOp2Reg(forop));   \
  }
      BINDCOMBDO(tor::AddIOp, hec::AddIOp)
      BINDCOMBDO(tor::SubIOp, hec::SubIOp)
      BINDCOMBDO(AndOp, hec::AndOp)
      BINDCOMBDO(OrOp, hec::OrOp)
      BINDCOMBDO(XOrOp, hec::XOrOp)
      BINDCOMBDO(ShiftLeftOp, hec::ShiftLeftOp)
      BINDCOMBDO(SignedShiftRightOp, hec::SignedShiftRightOp)
#undef BINDCOMBDO

#define BINDCOMBUDO(OpType, NewType)                                           \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op)) {                            \
    insertCombUnaryOp<OpType, NewType>(state0, sop, node.val0);                \
    insertCombUnaryOp<OpType, NewType>(state_entry, sop, mapOp2Reg(forop));    \
  }
      BINDCOMBUDO(NegFOp, hec::NegFOp)
      BINDCOMBUDO(TruncateIOp, hec::TruncateIOp)
      BINDCOMBUDO(SignExtendIOp, hec::SignExtendIOp)
#undef BINDCOMBUDO

      if (auto sop = llvm::dyn_cast<SelectOp>(opoe.op)) {
        insertSelectOp(state0, sop, node.val0);
        insertSelectOp(state_entry, sop, mapOp2Reg(forop));
      }

#define BINDMULTICYCLEUDO(OpType)                                              \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op)) {                            \
    insertMultiCycleUnaryOp(state0, state0_next, sop, node.val0, 0,            \
                            mapOp2Reg(forop));                                 \
    insertMultiCycleUnaryOp(state_entry, state_entry_next, sop,                \
                            mapOp2Reg(forop));                                 \
  }
      BINDMULTICYCLEUDO(SIToFPOp)
      BINDMULTICYCLEUDO(FPToSIOp)
#undef BINDMULTICYCLEUDO

#define BINDMULTICYCLEDO(OpType)                                               \
  if (auto sop = llvm::dyn_cast<OpType>(opoe.op)) {                            \
    insertMultiCycleBinaryOp(state0, state0_next, sop, node.val0, 0,           \
                             mapOp2Reg(forop));                                \
    insertMultiCycleBinaryOp(state_entry, state_entry_next, sop,               \
                             mapOp2Reg(forop));                                \
  }
      BINDMULTICYCLEDO(tor::MulIOp)
      BINDMULTICYCLEDO(tor::AddFOp)
      BINDMULTICYCLEDO(tor::SubFOp)
      BINDMULTICYCLEDO(tor::MulFOp)
      BINDMULTICYCLEDO(tor::DivFOp)
      BINDMULTICYCLEDO(SignedDivIOp)

#undef BINDMULTICYCLEDO

      if (auto cmpf = llvm::dyn_cast<tor::CmpFOp>(opoe.op)) {
        insertMultiCycleBinaryOp(state0, state0_next, cmpf, node.val0, 0,
                                 mapOp2Reg(forop));
        insertMultiCycleBinaryOp(state_entry, state_entry_next, cmpf,
                                 mapOp2Reg(forop));
      }

      if (auto cmpi = llvm::dyn_cast<tor::CmpIOp>(opoe.op)) {
        insertCmpIOp(state0, cmpi, node.val0, 0,
                     tor::stringifyEnum(cmpi.predicate()).str());
        insertCmpIOp(state_entry, cmpi, mapOp2Reg(forop), 0,
                     tor::stringifyEnum(cmpi.predicate()).str());
      }
      if (auto load = llvm::dyn_cast<tor::LoadOp>(opoe.op)) {
        insertLoadOp(state0, state0_next, load, node.val0, 0, mapOp2Reg(forop));
        insertLoadOp(state_entry, state_entry_next, load, mapOp2Reg(forop));
      }
      if (auto store = llvm::dyn_cast<tor::StoreOp>(opoe.op)) {
        insertStoreOp(state0, state0_next, store, node.val0, 0,
                      mapOp2Reg(forop));
        insertStoreOp(state_entry, state_entry_next, store, mapOp2Reg(forop));
      }
    }

    // Set yield
    tor::YieldOp yield =
        llvm::dyn_cast<tor::YieldOp>(forop.region().back().getTerminator());

    auto ptrItr = forop.getRegionIterArgs().begin();

    yield.dump();

    for (auto operand : yield.getOperands()) {
      std::cerr << "!!!!!" << std::endl;
      (*ptrItr).dump();

      rewriter.setInsertionPoint(&state_do_end.getBody()->back());
      rewriter.create<hec::AssignOp>(state_do_end.getLoc(),
                                     mapValue(*ptrItr++, node.tthen),
                                     mapValue(operand, node.tthen), nullptr);
    }

    // Update i, cond

    mlir::Type iType = forop.getInductionVar().getType();
    rewriter.setInsertionPoint(&state_do_end.getBody()->back());
    auto iAddStep = rewriter.create<hec::AddIOp>(
        state_do_end.getLoc(), iType,
        mapValue(forop.getInductionVar(), node.tthen),
        mapValue(forop.step(), node.tthen), nullptr);
    rewriter.setInsertionPointAfter(iAddStep);
    auto iassign = rewriter.create<hec::AssignOp>(
        state_do_end.getLoc(), mapValue(forop.getInductionVar(), node.tthen),
        iAddStep.res(), nullptr);
    rewriter.setInsertionPointAfter(iassign);
    auto iLessThan = rewriter.create<hec::CmpIOp>(
        state_do_end.getLoc(), rewriter.getI1Type(), iAddStep.res(),
        mapValue(forop.upperBound(), node.tthen), rewriter.getStringAttr("sle"),
        nullptr);
    rewriter.setInsertionPointAfter(iLessThan);
    rewriter.create<hec::AssignOp>(state_do_end.getLoc(), mapOp2Reg(forop),
                                   iLessThan.res(), nullptr);
    // Set exit
    ptrItr = forop.getRegionIterArgs().begin();
    for (auto result : forop.getResults()) {
      rewriter.setInsertionPoint(&state_entry.getBody()->back());
      rewriter.create<hec::AssignOp>(state_entry.getLoc(), mapValue(result, t),
                                     mapValue(*ptrItr++, t), node.val3);
    }
    auto initialPtrItr = forop.getIterOperands().begin();
    for (auto result : forop.getResults()) {
      rewriter.setInsertionPoint(&state0.getBody()->back());
      rewriter.create<hec::AssignOp>(state0.getLoc(), mapValue(result, t),
                                     mapValue(*initialPtrItr++, t), node.val1);
    }
  }

  auto insertPipelineForOp(size_t t) {}

  void gen_ops() {
    // Generate regs for argument
    if (state_count > 1)
      gen_argument_backup();

    std::cerr << "OpGen" << std::endl;

    // component.dump();

    for (size_t i = 0; i < nodes.size(); i++) {
      auto &node = nodes.at(topo[i]);
      std::cerr << "Node " << i << " ----" << std::endl;
      switch (node.type) {
      case TimeNode::NodeT::SEQ:
        for (auto opOnEdge : node.opsOnEdge) {
          insertSeqOp(opOnEdge.op, opOnEdge.from, opOnEdge.to);
        }
        break;
      case TimeNode::NodeT::CALL:
        insertCallOp(i);
        break;
      case TimeNode::NodeT::IF:
        insertIfOp(i);
        break;
      case TimeNode::NodeT::WHILE:
        insertWhileOp(i);
        break;
      case TimeNode::NodeT::FOR:
        if (node.getPipelineII() == -1)
          insertForOp(i);
        else
          insertPipelineForOp(i);
        break;
      default:
        break;
      }
    }
  }

  void gen_done(size_t t_end) {
    auto returnop =
        llvm::dyn_cast<tor::ReturnOp>(func.getBodyBlock()->getTerminator());
    auto &state_done = str2State[std::to_string(node2State[t_end])];
    auto trans =
        llvm::dyn_cast<hec::TransitionOp>(state_done.getBody()->back());
    rewriter.setInsertionPointToStart(trans.getBody());

    llvm::SmallVector<mlir::Value, 2> rets;
    for (auto res : returnop.getOperands())
      rets.push_back(mapValue(res, t_end));

    rewriter.create<hec::DoneOp>(trans.getLoc(), rets);
  }

public:
  STG(mlir::Operation *op, std::vector<TimeNode> &nodes,
      GlobalStorage &glbStorage,
      std::map<std::string, llvm::SmallVector<mlir::hec::ComponentPortInfo, 4>>
          dict,
      mlir::PatternRewriter &rewriter)
      : nodes(nodes), glbStorage(glbStorage), portDict(dict),
        rewriter(rewriter) {
    func = llvm::dyn_cast<tor::FuncOp>(op);
    assert(func != nullptr && "tor.FuncOp must exists");

    auto pipelineAttr = func->getAttrOfType<mlir::StringAttr>("pipeline");
    if (pipelineAttr == nullptr)
      style = Style::NORMAL;
    else if (pipelineAttr.getValue() == "for")
      style = Style::PIPELINEFOR;
    else
      style = Style::PIPELINEFUNC;

    set_topo(0, nodes.size() - 1);

    // func.dump();

    std::cerr << "Topo: ";
    for (auto x : topo)
      std::cerr << x << " ";
    std::cerr << std::endl;

    switch (style) {
    case Style::PIPELINEFOR:
      set_cycle();
      break;
    case Style::PIPELINEFUNC:
      set_cycle();
      break;
    default:
      break;
    }
  }

  unsigned gen_stages() {
    for (size_t i = 0, nCycle = cycles.size(); i < nCycle; i++) {
      rewriter.setInsertionPointToEnd(stageset.getBody());
      cycles.at(i).stageop = rewriter.create<hec::StageOp>(
          stageset.getLoc(),
          rewriter.getStringAttr(std::string("s") + std::to_string(i)));
    }
    // if (style == Style::PIPELINEFOR) {
    //   auto forop = llvm::dyn_cast<tor::ForOp>(nodes[0].op);
    //   for (auto operand : forop.getBody()->getArguments()) {
    //     rewriter.setInsertionPoint(stageset);
    //     auto wire =
    //         rewriter.create<hec::WireOp>(component.getLoc(),
    //         operand.getType());
    //     auto &vud = getVUD(operand, forop);
    //     vud.wire = wire;
    //   }
    // }
    return cycles.size() - 1;
  }

  void set_guards(unsigned t, unsigned tend,
                  std::vector<std::pair<mlir::Value, bool>> &guardVec,
                  bool reach = 1) {
    // std::cerr << t << ": " << guardVec.size() << std::endl;

    if (t == tend) {
      if (reach)
        guards[t] = guardVec;
      return;
    } else
      guards[t] = guardVec;

    if (nodes[t].type == TimeNode::NodeT::SEQ)
      set_guards(nodes[t].edges[0].to, tend, guardVec, reach);
    else if (nodes[t].type == TimeNode::NodeT::IF) {
      set_guards(nodes[t].tend, tend, guardVec, reach);
      auto ifop = llvm::dyn_cast<tor::IfOp>(nodes[t].op);
      guardVec.push_back(std::make_pair(ifop.condition(), 0));
      set_guards(nodes[t].edges[0].to, nodes[t].tend, guardVec, 0);
      guardVec.pop_back();
      if (nodes[t].edges.size() == 2) {
        guardVec.push_back(std::make_pair(ifop.condition(), 1));
        set_guards(nodes[t].edges[1].to, nodes[t].tend, guardVec, 0);
        guardVec.pop_back();
      }
    } else if (nodes[t].type == TimeNode::NodeT::FOR) {
      set_guards(nodes[t].edges[0].to, nodes[t].tend, guardVec, reach);
    }
  }

  mlir::Value map_value_stage(mlir::Value value, unsigned stage) {
    std::cerr << "map_value_stage on stage " << stage << " :";
    value.dump();
    mlir::Value ret;
    for (auto &vud : this->valueUseDefs)
      if (vud.value == value) {
        std::cerr << "\tfound vud : " << vud.id << std::endl;
        std::cerr << "\tpipelineRegs's size is " << pipelineRegs[vud.id].size()
                  << std::endl;
        if (pipelineRegs[vud.id].count(stage) == 0) {
          std::cerr << "\tregister not found:  "
                    << (pipelineRegs[vud.id].empty()
                            ? "pipelineReg is empty"
                            : std::to_string(stage) + " out of [" +
                                  std::to_string(
                                      pipelineRegs[vud.id].begin()->first) +
                                  ", " +
                                  std::to_string(
                                      pipelineRegs[vud.id].rbegin()->first) +
                                  "]")
                    << std::endl;
          if (vud.type == ValueUseDef::Type::Argument) {
            // for (auto pr : arg2arg)
            //   if (pr.first == value)
            //     ret = pr.second;
            ret = vud.wire;
          } else if (vud.type == ValueUseDef::Type::Variable) {
            ret = vud.wire;
          } else if (vud.type == ValueUseDef::Type::Constant) {
            ret = vud.owner->getResults().front();
          } else {
            ret = nullptr;
          }
        } else {
          std::cerr << "\tregister found "; // << std::endl;
          ret = registers[pipelineRegs[vud.id][stage]].op.getResult(0);
          ret.dump();
        }
        break;
      }
    if (ret == nullptr) {
      auto glbValue = glbStorage.getConstant(value);
      if (glbValue != nullptr)
        ret = glbValue;
    }
    return ret;
  }

  mlir::Value gen_guard(unsigned from, unsigned to, unsigned stage) {
    std::cerr << "gen_guard for " << from << " -> " << to << " on stage "
              << stage << std::endl;
    auto stageop = cycles[stage].stageop;
    auto get_value_stage = [&](mlir::Value value, bool rev) {
      mlir::Value ret = map_value_stage(value, stage);
      if (rev) {
        bool found = false;
        for (auto &op : stageop.getOps())
          if (auto notop = llvm::dyn_cast<hec::NotOp>(op)) {
            if (notop.src() == ret)
              ret = notop.getResult();
            found = true;
            break;
          }
        if (!found) {
          rewriter.setInsertionPointToEnd(stageop.getBody());
          auto notop = rewriter.create<hec::NotOp>(
              stageop.getLoc(), rewriter.getI1Type(), ret, nullptr);
          ret = notop.getResult();
        }
      }
      return ret;
    };

    auto get_and_value = [&](mlir::Value lhs, mlir::Value rhs) {
      mlir::Value ret = nullptr;
      for (auto &op : stageop.getOps()) {
        if (auto andop = llvm::dyn_cast<hec::AndOp>(op)) {
          if ((andop.lhs() == lhs && andop.rhs() == rhs) ||
              (andop.rhs() == lhs && andop.lhs() == rhs))
            ret = andop.getResult();
        }
      }
      if (ret == nullptr) {
        rewriter.setInsertionPointToEnd(stageop.getBody());
        auto andop = rewriter.create<hec::AndOp>(
            stageop.getLoc(), rewriter.getI1Type(), lhs, rhs, nullptr);
        ret = andop.getResult();
      }
      return ret;
    };

    auto guardVec = guards[from];
    for (auto pr : guardVec) {
      std::cerr << "\t";
      pr.first.dump();
    }
    if (nodes[from].type == TimeNode::NodeT::IF) {
      auto ifop = llvm::dyn_cast<tor::IfOp>(nodes[from].op);
      guardVec.push_back(
          std::make_pair(ifop.condition(), !(nodes[from].edges[0].to == to)));
    }

    mlir::Value guard;
    if (guardVec.empty()) {
      std::cerr << "no guard" << std::endl;
      guard = nullptr;
    } else {
      guard = get_value_stage(guardVec.front().first, guardVec.front().second);
      for (auto ptr = guardVec.begin() + 1; ptr != guardVec.end(); ptr++) {
        mlir::Value next = get_value_stage(ptr->first, ptr->second);
        guard = get_and_value(guard, next);
      }
    }
    return guard;
  }

  template <typename OldType, typename NewType>
  void gen_CombUnaryOp_on_stage(unsigned from, unsigned to, OldType op,
                                unsigned startstage, unsigned endstage) {
    assert(startstage == endstage);
    auto guard = gen_guard(from, to, startstage);
    auto lhs = map_value_stage(op->getOperand(0), startstage);
    auto res = map_value_stage(op->getResult(0), endstage + 1);
    assert(lhs != nullptr);
    // assert(res != nullptr);
    auto stageop = cycles[startstage].stageop;
    rewriter.setInsertionPointToEnd(stageop.getBody());
    auto new_op = rewriter.create<NewType>(
        stageop.getLoc(), op.getResult().getType(), lhs, guard);
    getVUD(op.getResult(), op).wire = new_op.getResult();

    if (res != nullptr) {
      rewriter.setInsertionPointToEnd(stageop.getBody());
      rewriter.create<hec::AssignOp>(stageop.getLoc(), res, new_op.getResult(),
                                     guard);
    }
  }

  template <typename OldType, typename NewType>
  void gen_CombOp_on_stage(unsigned from, unsigned to, OldType op,
                           unsigned startstage, unsigned endstage) {
    assert(startstage == endstage);
    auto guard = gen_guard(from, to, startstage);
    auto lhs = map_value_stage(op.lhs(), startstage);
    auto rhs = map_value_stage(op.rhs(), startstage);
    auto res = map_value_stage(op.result(), endstage + 1);
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    //   assert(res != nullptr);
    auto stageop = cycles[startstage].stageop;
    rewriter.setInsertionPointToEnd(stageop.getBody());
    auto new_op = rewriter.create<NewType>(
        stageop.getLoc(), op.getResult().getType(), lhs, rhs, guard);
    getVUD(op.getResult(), op).wire = new_op.getResult();

    if (res != nullptr) {
      rewriter.setInsertionPointToEnd(stageop.getBody());
      rewriter.create<hec::AssignOp>(stageop.getLoc(), res, new_op.getResult(),
                                     guard);
    }
  }

  void gen_CmpIOp_on_stage(unsigned from, unsigned to, tor::CmpIOp op,
                           unsigned startstage, unsigned endstage) {
    assert(startstage == endstage);
    auto guard = gen_guard(from, to, startstage);
    auto lhs = map_value_stage(op.lhs(), startstage);
    auto rhs = map_value_stage(op.rhs(), startstage);
    auto res = map_value_stage(op.result(), endstage + 1);
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    //   assert(res != nullptr);
    auto stageop = cycles[startstage].stageop;
    rewriter.setInsertionPointToEnd(stageop.getBody());

    auto new_op = rewriter.create<hec::CmpIOp>(
        stageop.getLoc(), op.getResult().getType(), lhs, rhs,
        tor::stringifyEnum(op.predicate()), guard);
    getVUD(op.getResult(), op).wire = new_op.getResult();

    if (res != nullptr) {
      rewriter.setInsertionPointToEnd(stageop.getBody());
      rewriter.create<hec::AssignOp>(stageop.getLoc(), res, new_op.getResult(),
                                     guard);
    }
  }

  void gen_SelectOp_on_stage(unsigned from, unsigned to, SelectOp op,
                             unsigned startstage, unsigned endstage) {
    assert(startstage == endstage);
    auto guard = gen_guard(from, to, startstage);
    auto condition = map_value_stage(op.condition(), startstage);
    auto lhs = map_value_stage(op.true_value(), startstage);
    auto rhs = map_value_stage(op.false_value(), startstage);
    auto res = map_value_stage(op.result(), endstage + 1);
    assert(condition != nullptr);
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    //   assert(res != nullptr);
    auto stageop = cycles[startstage].stageop;
    rewriter.setInsertionPointToEnd(stageop.getBody());

    auto new_op = rewriter.create<hec::SelectOp>(
        stageop.getLoc(), op.getResult().getType(), condition, lhs, rhs, guard);
    getVUD(op.getResult(), op).wire = new_op.getResult();

    if (res != nullptr) {
      rewriter.setInsertionPointToEnd(stageop.getBody());
      rewriter.create<hec::AssignOp>(stageop.getLoc(), res, new_op.getResult(),
                                     guard);
    }
  }

  template <typename OpType>
  void gen_MultiCycleUnaryOp_on_stage(unsigned from, unsigned to, OpType op,
                                      unsigned startstage, unsigned endstage) {
    op->dump();
    std::cerr << startstage << ", " << endstage << std::endl;

    assert(startstage < endstage);
    auto guard0 = gen_guard(from, to, startstage);
    auto guard1 = gen_guard(from, to, endstage);
    auto lhs = map_value_stage(op.in(), startstage);
    auto res = map_value_stage(op->getResult(0), endstage + 1);
    assert(lhs != nullptr);
    // assert(res != nullptr);

    auto primitive = getCellByOp(op).primitive;

    auto stageop0 = cycles[startstage].stageop;
    auto stageop1 = cycles[endstage].stageop;
    rewriter.setInsertionPointToEnd(stageop0.getBody());
    rewriter.create<hec::AssignOp>(stageop0.getLoc(), primitive.getResult(0),
                                   lhs, guard0);

    if (res != nullptr) {
      rewriter.setInsertionPointToEnd(stageop1.getBody());
      rewriter.create<hec::AssignOp>(stageop1.getLoc(), res,
                                     primitive.getResult(1), guard1);
    }

    getVUD(op->getResult(0), op).wire = primitive.getResult(1);
  }

  template <typename OpType>
  void gen_MultiCycleOp_on_stage(unsigned from, unsigned to, OpType op,
                                 unsigned startstage, unsigned endstage) {
    op->dump();
    std::cerr << startstage << ", " << endstage << std::endl;

    assert(startstage < endstage);
    auto guard0 = gen_guard(from, to, startstage);
    auto guard1 = gen_guard(from, to, endstage);
    auto lhs = map_value_stage(op.lhs(), startstage);
    auto rhs = map_value_stage(op.rhs(), startstage);
    auto res = map_value_stage(op.result(), endstage + 1);
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    // assert(res != nullptr);

    auto primitive = getCellByOp(op).primitive;

    auto stageop0 = cycles[startstage].stageop;
    auto stageop1 = cycles[endstage].stageop;
    rewriter.setInsertionPointToEnd(stageop0.getBody());
    rewriter.create<hec::AssignOp>(stageop0.getLoc(), primitive.getResult(0),
                                   lhs, guard0);
    rewriter.setInsertionPointToEnd(stageop0.getBody());
    rewriter.create<hec::AssignOp>(stageop0.getLoc(), primitive.getResult(1),
                                   rhs, guard0);

    if (res != nullptr) {
      rewriter.setInsertionPointToEnd(stageop1.getBody());
      rewriter.create<hec::AssignOp>(stageop1.getLoc(), res,
                                     primitive.getResult(2), guard1);
    }

    getVUD(op.result(), op).wire = primitive.getResult(2);
  }

  void gen_LoadOp_on_stage(unsigned from, unsigned to, tor::LoadOp load,
                           unsigned startstage, unsigned endstage) {
    assert(startstage < endstage);
    auto guard0 = gen_guard(from, to, startstage);
    auto guard1 = gen_guard(from, to, endstage);

    auto mem = mapValueMem(load.memref());

    assert(mem.id != -1ul);
    auto indices = load.indices();
    assert(indices.size() == 1 && "Require 1 indice for LoadOp");

    auto address = mem.getAddress();
    auto r_en = mem.getReadEnable();
    auto r_data = mem.getReadData();

    auto indice = map_value_stage(indices.front(), startstage);

    auto res = map_value_stage(load.getResult(), endstage + 1);

    assert(indice != nullptr);
    assert(address != nullptr);
    assert(r_en != nullptr);
    assert(r_data != nullptr);
    // assert(res != nullptr);

    auto stageop0 = cycles[startstage].stageop;
    auto stageop1 = cycles[endstage].stageop;

    rewriter.setInsertionPointToEnd(stageop0.getBody());
    rewriter.create<mlir::hec::AssignOp>(stageop0.getLoc(), address, indice,
                                         guard0);
    rewriter.setInsertionPointToEnd(stageop0.getBody());
    rewriter.create<mlir::hec::EnableOp>(stageop0.getLoc(), r_en, guard0);

    if (res != nullptr) {
      rewriter.setInsertionPointToEnd(stageop1.getBody());
      rewriter.create<mlir::hec::AssignOp>(stageop1.getLoc(), res, r_data,
                                           guard1);
    }
    getVUD(load.result(), load).wire = r_data;
  }

  void gen_StoreOp_on_stage(unsigned from, unsigned to, tor::StoreOp store,
                            unsigned startstage, unsigned endstage) {
    assert(startstage < endstage);
    auto guard0 = gen_guard(from, to, startstage);
    // auto guard1 = gen_guard(from, to, endstage);

    auto mem = mapValueMem(store.memref());

    assert(mem.id != -1ul);
    auto indices = store.indices();
    assert(indices.size() == 1 && "Require 1 indice for StoreOp");

    auto address = mem.getAddress();
    auto w_en = mem.getWriteEnable();
    auto w_data = mem.getWriteData();

    auto indice = map_value_stage(indices.front(), startstage);

    auto operand = map_value_stage(store.value(), startstage);

    assert(indice != nullptr && address != nullptr && w_en != nullptr &&
           w_data != nullptr);
    assert(operand != nullptr);

    auto stageop0 = cycles[startstage].stageop;
    // auto stageop1 = cycles[endstage].stageop;

    rewriter.setInsertionPointToEnd(stageop0.getBody());
    rewriter.create<mlir::hec::AssignOp>(stageop0.getLoc(), address, indice,
                                         guard0);
    rewriter.setInsertionPointToEnd(stageop0.getBody());
    rewriter.create<mlir::hec::AssignOp>(stageop0.getLoc(), w_data, operand,
                                         guard0);
    rewriter.setInsertionPointToEnd(stageop0.getBody());
    rewriter.create<mlir::hec::EnableOp>(stageop0.getLoc(), w_en, guard0);
  }

  void gen_op_on_stage(unsigned from, unsigned to, mlir::Operation *op,
                       unsigned startstage, unsigned endstage) {
    std::cerr << "gen_op_on_stage (" << startstage << ", " << endstage << "): ";
    op->dump();

#define GENCOMB(OldType, NewType)                                              \
  if (auto sop = llvm::dyn_cast<OldType>(op))                                  \
    gen_CombOp_on_stage<OldType, NewType>(from, to, sop, startstage,           \
                                          endstage - 1);

    GENCOMB(tor::AddIOp, hec::AddIOp)
    GENCOMB(tor::SubIOp, hec::SubIOp)
    GENCOMB(AndOp, hec::AndOp)
    GENCOMB(OrOp, hec::OrOp)
    GENCOMB(XOrOp, hec::XOrOp)
    GENCOMB(ShiftLeftOp, hec::ShiftLeftOp)
    GENCOMB(SignedShiftRightOp, hec::SignedShiftRightOp)
#undef GENCOMB

#define GENCOMBU(OldType, NewType)                                             \
  if (auto sop = llvm::dyn_cast<OldType>(op))                                  \
    gen_CombUnaryOp_on_stage<OldType, NewType>(from, to, sop, startstage,      \
                                               endstage - 1);

    GENCOMBU(NegFOp, hec::NegFOp)
    GENCOMBU(TruncateIOp, hec::TruncateIOp)
    GENCOMBU(SignExtendIOp, hec::SignExtendIOp)
#undef GENCOMBU

    if (auto sop = llvm::dyn_cast<SelectOp>(op))
      gen_SelectOp_on_stage(from, to, sop, startstage, endstage - 1);

    if (auto cmpi = llvm::dyn_cast<tor::CmpIOp>(op)) {
      gen_CmpIOp_on_stage(from, to, cmpi, startstage, endstage - 1);
    }

#define GENMULTICYCLEU(OpType)                                                 \
  if (auto sop = llvm::dyn_cast<OpType>(op))                                   \
    gen_MultiCycleUnaryOp_on_stage(from, to, sop, startstage, endstage);
    GENMULTICYCLEU(SIToFPOp)
    GENMULTICYCLEU(FPToSIOp)

#undef GENMULTICYCLEU

#define GENMULTICYCLE(OpType)                                                  \
  if (auto sop = llvm::dyn_cast<OpType>(op))                                   \
    gen_MultiCycleOp_on_stage(from, to, sop, startstage, endstage);

    GENMULTICYCLE(tor::AddFOp)
    GENMULTICYCLE(tor::SubFOp)
    GENMULTICYCLE(tor::MulIOp)
    GENMULTICYCLE(tor::MulFOp)
    GENMULTICYCLE(tor::DivFOp)
    GENMULTICYCLE(SignedDivIOp)
    GENMULTICYCLE(tor::CmpFOp)
#undef GENMULTICYCLE

    if (auto load = llvm::dyn_cast<tor::LoadOp>(op))
      gen_LoadOp_on_stage(from, to, load, startstage, endstage);
    if (auto store = llvm::dyn_cast<tor::StoreOp>(op))
      gen_StoreOp_on_stage(from, to, store, startstage, endstage);
  }

  void gen_ops_pipeline() {
    std::vector<std::pair<mlir::Value, bool>> guardVec;
    guards.resize(nodes.size());
    if (style == Style::PIPELINEFUNC)
      set_guards(0, nodes.size() - 1, guardVec, 1);
    else
      set_guards(0, nodes[0].tend, guardVec, 1);

    // std::cerr << "!!!!!" << std::endl;
    // for (size_t i = 0; i < nodes.size(); i++) {
    //   std::cerr << i << " has " << guards[i].size() << " guards" <<
    //   std::endl;
    // }

    for (unsigned i = 0, n = style == Style::PIPELINEFUNC ? nodes.size() - 1
                                                          : nodes[0].tend;
         i <= n; i++) {
      // std::cerr << "Guard of " << i << ": ";
      // for (auto guard : guards[i])
      //   std::cerr << guard.second << " ";
      // std::cerr << std::endl;
    }

    auto findVUD = [&](mlir::Value res) -> ValueUseDef {
      for (auto x : valueUseDefs)
        if (x.value == res)
          return x;
      assert("not found");
      return ValueUseDef();
    };

    if (style == Style::PIPELINEFOR) {
      rewriter.setInsertionPoint(stageset);

      auto wire = rewriter.create<hec::WireOp>(
          component.getLoc(), component.getArgument(0).getType(),
          rewriter.getStringAttr("i"));
      inductionWire = wire.getResult();

      auto forop = llvm::dyn_cast<tor::ForOp>(nodes[0].op);

      for (auto &x : valueUseDefs)
        if (x.value == forop.getInductionVar())
          x.wire = inductionWire;

      for (unsigned j = 0; j < component.numInPorts() - 1; j++) {
        component.getArgument(j).dump();
        std::cerr << "the id for argument " << j << " is "
                  << findVUD(func.getArgument(j)).id << std::endl;
        rewriter.setInsertionPoint(stageset);
        rewriter.create<hec::InitOp>(
            component.getLoc(),
            registers
                [pipelineRegs[findVUD(func.getArgument(j)).id].begin()->second]
                    .op.getResult(0),
            component.getArgument(j));

        for (auto &vud : valueUseDefs)
          if (vud.value == func.getArgument(j))
            vud.wire = registers[pipelineRegs[findVUD(func.getArgument(j)).id]
                                     .begin()
                                     ->second]
                           .op.getResult(0);
      }

      size_t i = 0;
      for (auto arg : forop.getRegionIterArgs()) {
        rewriter.setInsertionPoint(stageset);

        if (forop.initArgs()[i].isa<BlockArgument>()) {
          rewriter.create<hec::InitOp>(
              component.getLoc(), // reg.getResult(0),
              registers[pipelineRegs[findVUD(arg).id].begin()->second]
                  .op.getResult(0),
              component.getArgument(
                  forop.initArgs()[i].cast<BlockArgument>().getArgNumber()));
        } else {
          rewriter.create<hec::InitOp>(
              component.getLoc(), // reg.getResult(0),
              registers[pipelineRegs[findVUD(arg).id].begin()->second]
                  .op.getResult(0),
              glbStorage.getConstant(forop.initArgs()[i]));
        }
        i++;
      }
    } else {
      for (unsigned j = 0; j < component.numInPorts() - 1; j++) {
        component.getArgument(j).dump();
        std::cerr << "the id for argument " << j << " is "
                  << findVUD(func.getArgument(j)).id << std::endl;
        rewriter.setInsertionPoint(stageset);
        rewriter.create<hec::InitOp>(
            component.getLoc(),
            registers
                [pipelineRegs[findVUD(func.getArgument(j)).id].begin()->second]
                    .op.getResult(0),
            component.getArgument(j));

        for (auto &vud : valueUseDefs)
          if (vud.value == func.getArgument(j))
            vud.wire = registers[pipelineRegs[findVUD(func.getArgument(j)).id]
                                     .begin()
                                     ->second]
                           .op.getResult(0);
      }
    }
    for (size_t i = 0, nCycle = cycles.size(); i < nCycle; i++) {
      auto &cycle = cycles.at(i);
      for (auto nodeid : cycle.nodeIds) {
        auto &node = nodes.at(nodeid);

        for (auto opoe : node.opsOnEdge) {
          gen_op_on_stage(opoe.from, opoe.to, opoe.op, depth[opoe.from],
                          depth[opoe.to]);
        }

        if (node.type == TimeNode::NodeT::IF) {
          std::cerr << "meet if-end";
          auto stage = depth[node.tend];
          std::cerr << " put it at stage " << stage - 1 << std::endl;
          cycles[stage - 1].reviewIds.push_back(nodeid);
        }
      }

      for (auto nodeid : cycle.reviewIds) {
        auto &node = nodes.at(nodeid);
        assert(node.type == TimeNode::NodeT::IF);
        auto stage = depth[node.tend];
        std::cerr << "process if at stage " << stage - 1 << std::endl;
        auto stageop = cycles[stage - 1].stageop;
        auto ifop = llvm::dyn_cast<tor::IfOp>(node.op);
        auto &node_end = nodes.at(node.tend);
        auto yield_then = llvm::dyn_cast<tor::YieldOp>(
            ifop.thenRegion().front().getTerminator());

        auto ptr = yield_then.results().begin();
        for (auto res : ifop.results()) {
          rewriter.setInsertionPointToEnd(stageop.getBody());
          auto guard =
              gen_guard(node_end.backEdges[0].from, node.tend, stage - 1);
          std::cerr << "guard: ";
          guard.dump();

          auto dest = map_value_stage(res, stage);
          auto src = map_value_stage(*ptr, stage - 1);
          assert(dest != nullptr);
          assert(src != nullptr);
          rewriter.create<hec::AssignOp>(stageop.getLoc(), dest, src, guard);
        }

        if (node.edges.size() == 2) {
          auto yield_else = llvm::dyn_cast<tor::YieldOp>(
              ifop.elseRegion().front().getTerminator());
          auto ptr = yield_else.results().begin();
          for (auto res : ifop.results()) {
            rewriter.setInsertionPointToEnd(stageop.getBody());
            rewriter.create<hec::AssignOp>(
                stageop.getLoc(), map_value_stage(res, stage),
                map_value_stage(*ptr, stage - 1),
                gen_guard(node_end.backEdges[1].from, node.tend, stage - 1));
          }
        }
      }
    }

    if (style == Style::PIPELINEFUNC) {
      auto stage = depth[nodes.size() - 1];
      auto stageop = cycles[stage].stageop;
      auto returnop = llvm::dyn_cast<tor::ReturnOp>(
          func.getRegion().front().getTerminator());
      llvm::SmallVector<mlir::Value, 2> results;
      for (auto res : returnop.operands()) {
        results.push_back(map_value_stage(res, stage));
      }
      rewriter.setInsertionPointToEnd(stageop.getBody());
      rewriter.create<hec::YieldOp>(stageop.getLoc(),
                                    mlir::ValueRange(results));
    } else {
      std::cerr << "???" << std::endl;
      // component.dump();
      auto node = nodes.at(0);
      auto forop = llvm::dyn_cast<tor::ForOp>(node.op);
      auto yieldop = llvm::dyn_cast<tor::YieldOp>(
          forop.getRegion().back().getTerminator());

      auto retop = llvm::dyn_cast<tor::ReturnOp>(
          func.getRegion().back().getTerminator());

      std::map<size_t, size_t> res2ret;
      for (size_t i = 0; i < retop.operands().size(); i++) {
        auto value = retop.getOperand(i);
        auto op = value.getDefiningOp();
        if (auto forop = llvm::dyn_cast<tor::ForOp>(op)) {
          bool found = 0;
          for (size_t j = 0; j < forop.getResults().size(); j++)
            if (forop.getResult(j) == value) {
              found = 1;
              res2ret[j] = i;
            }
          assert(found);
        } else
          assert(0 && "return weird value");
      }

      for (auto pr : res2ret) {
        std::cerr << "return " << pr.first << " to " << pr.second << std::endl;
      }

      for (size_t i = 0; i < yieldop.results().size(); i++) {
        auto res = yieldop->getOperand(i);

        ValueUseDef vud = findVUD(res);
        ValueUseDef vud_ =
            findVUD(forop.getRegionIterArgs().take_front(i + 1).back());

        forop.getRegionIterArgs().take_front(i + 1).back().dump();

        auto stage = depth[vud.def];
#define COMB(type)                                                             \
  if (auto op = llvm::dyn_cast<type>(vud.owner))                               \
    stage -= 1;
        COMB(tor::AddIOp)
        COMB(tor::CmpIOp)
        COMB(AndOp)
        COMB(OrOp)
        COMB(XOrOp)
        COMB(ShiftLeftOp)
        COMB(SignedShiftRightOp)
        COMB(NegFOp)
        COMB(TruncateIOp)
        COMB(SignExtendIOp)
        COMB(SelectOp)
#undef COMB

        std::cerr << "stage is " << stage << std::endl;
        auto stageop = cycles[stage].stageop;

        stageop.dump();

        auto II = component->getAttrOfType<IntegerAttr>("II").getInt();

        if (auto ifop = dyn_cast<tor::IfOp>(vud.owner)) {
          rewriter.setInsertionPointToEnd(stageop.getBody());
          auto rid = pipelineRegs[vud.id][stage];
          rewriter.create<hec::DeliverOp>(
              stageop.getLoc(), registers[rid].op.getResult(0),
              map_value_stage(
                  forop.getRegionIterArgs().take_front(i + 1).back(),
                  stage + 1 > II ? stage + 1 - II : 1),
              (res2ret.find(i) != res2ret.end())
                  ? component.getArgument(component.numInPorts() + res2ret[i])
                  : component.getArgument(component.getNumArguments() - 1),
              nullptr);
        } else {
          auto rid = pipelineRegs[vud.id][stage + 1];
          for (auto &op : *stageop.getBody())
            if (auto assign = llvm::dyn_cast<hec::AssignOp>(op))
              if (assign.dest() == registers[rid].op->getResult(0)) {
                rewriter.setInsertionPointToEnd(stageop.getBody());
                rewriter.create<hec::DeliverOp>(
                    stageop.getLoc(), assign.src(),
                    map_value_stage(
                        forop.getRegionIterArgs().take_front(i + 1).back(),
                        stage + 1 > II ? stage + 1 - II : 1),
                    (res2ret.find(i) != res2ret.end())
                        ? component.getArgument(component.numInPorts() +
                                                res2ret[i])
                        : component.getArgument(component.getNumArguments() -
                                                1),
                    assign.guard());
              }
        }
      }
    }

    if (style == Style::PIPELINEFOR) {
      auto forop = llvm::dyn_cast<tor::ForOp>(nodes[0].op);
      auto vud = findVUD(forop.getInductionVar());
      auto ptr_next = pipelineRegs[vud.id].find(1);
      if (ptr_next != pipelineRegs[vud.id].end()) {
        auto stageop = cycles[0].stageop;
        rewriter.setInsertionPointToEnd(stageop.getBody());
        rewriter.create<hec::AssignOp>(
            stageop.getLoc(), registers[ptr_next->second].op.getResult(0),
            inductionWire, nullptr);
      }
    }
    // component.dump();

    llvm::SmallVector<hec::PrimitiveOp, 8> toErase;

    for (auto &op : *component.getBody())
      if (auto reg = llvm::dyn_cast<hec::PrimitiveOp>(op))
        if (reg.primitiveName() == "register" &&
            reg.getResult(0).getUses().empty()) {
          auto regname = reg.instanceName();
          // std::cerr << regname.str() << std::endl;
          auto stagestr = regname.rsplit('_').second.str();
          unsigned stage = std::stoi(stagestr);
          auto vudstr = regname.split('_').second.split('_').first.str();
          // std::cerr << vudstr << std::endl;
          auto vid = std::stoi(vudstr);

          if (pipelineRegs[vid].begin()->first != stage)
            continue;
          pipelineRegs[vid].erase(stage);
          toErase.push_back(reg);
          // rewriter.eraseOp(reg);
          // std::cerr << "erase ";
          // reg.dump();
        }

    for (auto op : toErase) {
      std::cerr << "erase ";
      op.dump();
      rewriter.eraseOp(op);
    }
    toErase.clear();

    for (auto vud : valueUseDefs)
      if (pipelineRegs[vud.id].size() >= 2) {
        auto ptr_next = pipelineRegs[vud.id].begin();
        auto ptr = ptr_next++;
        for (; ptr_next != pipelineRegs[vud.id].end();) {
          auto stage = ptr->first;
          auto stageop = cycles[stage].stageop;
          rewriter.setInsertionPointToEnd(stageop.getBody());
          rewriter.create<hec::AssignOp>(
              stageop.getLoc(), registers[ptr_next->second].op.getResult(0),
              registers[ptr->second].op.getResult(0), nullptr);
          ptr_next++;
          ptr++;
        }
      }

    for (auto &op : *component.getBody())
      if (auto reg = llvm::dyn_cast<hec::PrimitiveOp>(op)) {
        auto regname = reg.instanceName();
        auto stagestr = regname.rsplit('_').second.str();
        unsigned stage = std::stoi(stagestr);
        unsigned stageafter =
            std::stoi(cycles.rbegin()->stageop.getName().drop_front().str()) +
            1;
        if (stage >= stageafter) {
          for (auto user : reg.getResult(0).getUsers())
            rewriter.eraseOp(user);
          toErase.push_back(reg);
        }
      }

    for (auto op : toErase) {
      rewriter.eraseOp(op);
    }
    toErase.clear();

    for (auto cycleptr = cycles.rbegin(); cycleptr != cycles.rend();
         cycleptr++) {
      auto isEmpty = [&]() {
        auto stateop = cycleptr->stageop;
        return stateop.getBody()->getOperations().size() == 0;
      };

      if (isEmpty()) {
        unsigned stagetoerase =
            std::stoi(cycleptr->stageop.getName().drop_front().str());
        std::cerr << "Erase stage " << stagetoerase << std::endl;
        for (auto &op : *component.getBody())
          if (auto reg = llvm::dyn_cast<hec::PrimitiveOp>(op)) {
            auto regname = reg.instanceName();
            auto stagestr = regname.rsplit('_').second.str();
            unsigned stage = std::stoi(stagestr);
            if (stage == stagetoerase) {
              for (auto user : reg.getResult(0).getUsers())
                rewriter.eraseOp(user);
              toErase.push_back(reg);
            }
          }
        rewriter.eraseOp(cycleptr->stageop);
      } else
        break;
    }

    for (auto op : toErase) {
      rewriter.eraseOp(op);
    }
  }

  void codegen(mlir::hec::DesignOp design) {
    gen_component(design);

    set_regs();
    set_cells();

    gen_constants();
    gen_cells();
    gen_mems();

    unsigned t_end;
    if (style == Style::NORMAL) {
      gen_regs();
      gen_calls();
      t_end = gen_states(0, nodes.size() - 1);
      // for (auto pr : node2State)
      // std::cerr << "node " << pr.first << " -> state s" << pr.second
      // << std::endl;
      gen_ops();
      gen_done(t_end);
    } else {
      t_end = gen_stages();
      gen_ops_pipeline();
    }
    // component.dump();
  }
}; // namespace hecgen

class Component {
private:
  mlir::Operation *funcOp;
  llvm::StringRef symbol;
  std::vector<TimeNode> nodes;

  size_t ec, nc;
  // enum class Interfc { NAKED, WRAPPED } interfc;
  enum class Style { STG, GRAPH } style;

  GlobalStorage &glbStorage;
  std::map<std::string, llvm::SmallVector<mlir::hec::ComponentPortInfo, 4>>
      portDict;

  void dbg_print_timeGraph() {
    std::cerr << "Time Graph with " << nodes.size() << " nodes, " << ec
              << " edges" << std::endl;
    for (auto node : nodes) {
      std::cerr << "Node " << node.t << " is of type ";
      if (node.type == TimeNode::NodeT::SEQ)
        std::cerr << "SEQ";
      else if (node.type == TimeNode::NodeT::IF)
        std::cerr << "IF";
      else if (node.type == TimeNode::NodeT::FOR)
        std::cerr << "FOR";
      else if (node.type == TimeNode::NodeT::WHILE)
        std::cerr << "WHILE";

      std::cerr << " , tend is " << node.tend << ": ";
      std::cerr << "links to ";

      for (auto edge : node.edges) {
        std::cerr << "{";
        std::cerr << edge.from << "->" << edge.to << "} ";
      }

      std::cerr << std::endl;
    }

    std::cerr << "Op on edges:" << std::endl;
    for (auto node : nodes) {
      std::cerr << "starts at node " << node.t << " :";
      for (auto op : node.opsOnEdge) {
        std::cerr << " {" << op.from << "->" << op.to << ": "
                  << op.op->getName().getStringRef().str() << "}";
      }
      std::cerr << std::endl;
    }

    std::cerr << "Implemented in " << (style == Style::STG ? "STG" : "GRAPH")
              << " style" << std::endl;
  }

  void gen_STG(mlir::hec::DesignOp design, mlir::PatternRewriter &rewriter) {
    std::cerr << "the size of nodes is " << nodes.size() << std::endl;
    auto STGptr = new STG(funcOp, nodes, glbStorage, portDict, rewriter);
    STGptr->codegen(design);

    std::cerr << "## Generated STG for component " << symbol.str() << std::endl
              << std::endl;
  }

  // void gen_Graph(mlir::hec::DesignOp design, mlir::PatternRewriter &rewriter)
  // {}

public:
  Component(
      mlir::tor::FuncOp &func, GlobalStorage &storage,
      std::map<std::string, llvm::SmallVector<mlir::hec::ComponentPortInfo, 4>>
          portDict)
      : glbStorage(storage), portDict(portDict) {
    std::cerr << "Parse FuncOp : " << (std::string)func.getName() << std::endl;
    funcOp = func;
    symbol = func.getName();

    auto strategy = func->getAttrOfType<mlir::StringAttr>("strategy");
    assert(strategy != nullptr && "must be static or dynamic");

    // Load time graph
    // bool edynamic = false;
    // bool estatic = false;

    for (auto &op : func.body().front()) {
      if (llvm::isa<mlir::tor::TimeGraphOp>(op)) {
        auto tg = llvm::dyn_cast<mlir::tor::TimeGraphOp>(op);
        for (auto i = tg.starttime(); i <= tg.endtime(); i++)
          nodes.push_back(TimeNode(i));
        nc = tg.endtime() - tg.starttime() + 1;
        ec = 0;
        for (auto &op1 : tg.region().front())
          if (llvm::isa<mlir::tor::SuccTimeOp>(op1)) {
            auto succ = llvm::dyn_cast<mlir::tor::SuccTimeOp>(op1);
            auto to = succ.time();
            auto froms = succ.points();
            auto eattrarray = succ.edges();
            for (size_t i = 0; i < froms.size(); i++) {
              auto fromAttr = froms[i];
              auto edict = eattrarray[i].cast<mlir::DictionaryAttr>();
              auto format = edict.get("type");
              auto from = fromAttr.cast<mlir::IntegerAttr>().getInt();
              auto info = format.cast<mlir::StringAttr>().getValue().str();

              nodes[from].edges.push_back(
                  TimeEdge(from, to, edict,
                           info.find("dynamic") != StringRef::npos
                               ? TimeEdge::SD::DYNAMIC
                               : TimeEdge::SD::STATIC,
                           info.find("for") != StringRef::npos ||
                               info.find("while") != StringRef::npos));
              nodes[to].backEdges.push_back(nodes[from].edges.back());
              nodes[from].tend = to;
              ec++;
            }
          }
      }
    }

    /*
    assert(strategy != nullptr);
    if (strategy.getValue() == "dynamic") {
      assert(!estatic && "Can't contain dynamic edges");
      style = Style::GRAPH;
    } else if (strategy.getValue() == "static") {
      assert(!edynamic && "Can't contain dynamic edges");
      style = Style::STG;
    } else {
      assert(0 && "Unexpected style");
    }
    */

    style = Style::STG;
    // Bind operations to edges
    auto bind_operation = [&](uint64_t src, uint64_t dest,
                              mlir::Operation *op) {
      nodes[src].opsOnEdge.push_back({src, dest, op});
    };

    func.walk([&](mlir::Operation *op) {
#define BIND(OpType)                                                           \
  if (auto sop = llvm::dyn_cast<OpType>(op))                                   \
    bind_operation(sop.starttime(), sop.endtime(), op);
      BIND(tor::AddIOp)
      BIND(tor::SubIOp)
      BIND(tor::MulIOp)
      BIND(tor::CmpIOp)
      BIND(tor::AddFOp)
      BIND(tor::SubFOp)
      BIND(tor::MulFOp)
      BIND(tor::DivFOp)
      BIND(tor::CmpFOp)
      BIND(tor::LoadOp)
      BIND(tor::StoreOp)
#undef BIND

#define BINDSTD(OpType)                                                        \
  if (auto sop = llvm::dyn_cast<OpType>(op))                                   \
    bind_operation(sop->getAttrOfType<IntegerAttr>("starttime").getInt(),      \
                   sop->getAttrOfType<IntegerAttr>("endtime").getInt(), op);
      BINDSTD(AndOp);
      BINDSTD(OrOp);
      BINDSTD(XOrOp);
      BINDSTD(ShiftLeftOp);
      BINDSTD(NegFOp)
      BINDSTD(SignedShiftRightOp);
      BINDSTD(TruncateIOp);
      BINDSTD(SignExtendIOp);
      BINDSTD(SIToFPOp)
      BINDSTD(FPToSIOp)
      BINDSTD(SelectOp)
      BINDSTD(SignedDivIOp)
#undef BINDSTD

      if (auto callOp = llvm::dyn_cast<tor::CallOp>(op)) {
        nodes[callOp.starttime()].opsOnEdge.push_back(
            {callOp.starttime(), callOp.endtime(), callOp});
        assert(nodes[callOp.starttime()].op == nullptr);
        nodes[callOp.starttime()].op = callOp;
        nodes[callOp.starttime()].tend = callOp.endtime();
        nodes[callOp.starttime()].type = TimeNode::NodeT::CALL;
      }
      if (auto ifop = llvm::dyn_cast<tor::IfOp>(op)) {
        assert(nodes[ifop.starttime()].op == nullptr);
        nodes[ifop.starttime()].op = ifop;
        nodes[ifop.starttime()].tend = ifop.endtime();
        nodes[ifop.starttime()].type = TimeNode::NodeT::IF;
      } else if (auto whileOp = llvm::dyn_cast<tor::WhileOp>(op)) {
        nodes[whileOp.endtime()].op = whileOp;
        nodes[whileOp.endtime()].t = whileOp.starttime();
        nodes[whileOp.endtime()].type = TimeNode::NodeT::ENDWHILE;
        assert(nodes[whileOp.starttime()].op == nullptr);
        nodes[whileOp.starttime()].op = whileOp;
        nodes[whileOp.starttime()].tend = whileOp.endtime();
        nodes[whileOp.starttime()].type = TimeNode::NodeT::WHILE;
      } else if (auto forOp = llvm::dyn_cast<tor::ForOp>(op)) {
        nodes[forOp.endtime()].op = forOp;
        nodes[forOp.endtime()].t = forOp.starttime();
        nodes[forOp.endtime()].type = TimeNode::NodeT::ENDFOR;
        assert(nodes[forOp.starttime()].op == nullptr);
        nodes[forOp.starttime()].op = forOp;
        nodes[forOp.starttime()].tend = forOp.endtime();
        nodes[forOp.starttime()].type = TimeNode::NodeT::FOR;
      }
    });
    dbg_print_timeGraph();
  }

  void rewrite(mlir::hec::DesignOp design, mlir::PatternRewriter &rewriter) {
    std::cerr << std::endl
              << "Rewrite component for FuncOp " << (std::string)this->symbol
              << " in " << (style == Style::STG ? "STG" : "GRAPH") << " style"
              << std::endl;
    // if (style == Style::STG)
    gen_STG(design, rewriter);
    // else
    // gen_Graph(design, rewriter);
  }

  std::string getName() { return (std::string)this->symbol; }
};

void standardizeFunc(mlir::tor::FuncOp func, mlir::PatternRewriter &rewriter) {
  std::map<size_t, size_t> dict;
  size_t count = 0;
  auto add2Dict = [&](size_t node) {
    if (dict.find(node) == dict.end())
      dict[node] = count++;
  };

  for (auto &op : func.body().front())
    if (auto tg = llvm::dyn_cast<mlir::tor::TimeGraphOp>(op)) {
      auto starttime = tg.starttime();
      add2Dict(starttime);
      for (auto &op1 : tg.region().front())
        if (auto succ = llvm::dyn_cast<tor::SuccTimeOp>(op1)) {
          add2Dict(succ.time());
          auto froms = succ.points();
          for (auto attr : froms)
            add2Dict(attr.cast<mlir::IntegerAttr>().getInt());
        }

      tg.starttimeAttr(
          rewriter.getIntegerAttr(tg.starttimeAttr().getType(), 0));
      tg.endtimeAttr(
          rewriter.getIntegerAttr(tg.endtimeAttr().getType(), count - 1));

      for (auto &op1 : tg.region().front())
        if (auto succ = llvm::dyn_cast<tor::SuccTimeOp>(op1)) {
          auto to = succ.time();
          succ.timeAttr(
              rewriter.getIntegerAttr(succ.timeAttr().getType(), dict[to]));

          llvm::SmallVector<mlir::IntegerAttr, 2> arr;

          for (auto attr : succ.points())
            arr.push_back(rewriter.getIntegerAttr(
                attr.cast<mlir::IntegerAttr>().getType(),
                dict[attr.cast<mlir::IntegerAttr>().getInt()]));

          succ.pointsAttr(rewriter.getArrayAttr(
              llvm::ArrayRef<mlir::Attribute>(arr.begin(), arr.end())));
        }
    }

  func.walk([&](mlir::Operation *op) {
#define MODIFY(OpType)                                                         \
  if (auto sop = llvm::dyn_cast<OpType>(op)) {                                 \
    sop.starttimeAttr(rewriter.getIntegerAttr(sop.starttimeAttr().getType(),   \
                                              dict[sop.starttime()]));         \
    sop.endtimeAttr(rewriter.getIntegerAttr(sop.endtimeAttr().getType(),       \
                                            dict[sop.endtime()]));             \
  }
    MODIFY(tor::AddIOp)
    MODIFY(tor::SubIOp)
    MODIFY(tor::MulIOp)
    MODIFY(tor::CmpIOp)
    MODIFY(tor::AddFOp)
    MODIFY(tor::SubFOp)
    MODIFY(tor::MulFOp)
    MODIFY(tor::DivFOp)
    MODIFY(tor::CmpFOp)
    MODIFY(tor::LoadOp)
    MODIFY(tor::StoreOp)
    MODIFY(tor::WhileOp)
    MODIFY(tor::ForOp)
    MODIFY(tor::IfOp)
    MODIFY(tor::CallOp)
#undef MODIFY
#define MODIFYSTD(OpType)                                                      \
  if (auto sop = llvm::dyn_cast<OpType>(op)) {                                 \
    sop->setAttr(                                                              \
        "starttime",                                                           \
        rewriter.getIntegerAttr(                                               \
            sop->getAttr("starttime").getType(),                               \
            dict[sop->getAttrOfType<IntegerAttr>("starttime").getInt()]));     \
    sop->setAttr(                                                              \
        "endtime",                                                             \
        rewriter.getIntegerAttr(                                               \
            sop->getAttr("endtime").getType(),                                 \
            dict[sop->getAttrOfType<IntegerAttr>("endtime").getInt()]));       \
  }
    //    MODIFYSTD(CmpFOp)
    MODIFYSTD(AndOp)
    MODIFYSTD(OrOp)
    MODIFYSTD(XOrOp)
    MODIFYSTD(ShiftLeftOp)
    MODIFYSTD(SignedShiftRightOp)
    MODIFYSTD(NegFOp)
    MODIFYSTD(TruncateIOp)
    MODIFYSTD(SignExtendIOp)
    MODIFYSTD(SIToFPOp)
    MODIFYSTD(FPToSIOp)
    MODIFYSTD(SelectOp)
    MODIFYSTD(SignedDivIOp)
#undef MODIFYSTD
  });

  // func.dump();
}

int genComponents(mlir::tor::DesignOp design, mlir::PatternRewriter &rewriter,
                  GlobalStorage &glbStorage,
                  std::vector<hecgen::Component> &components) {
  std::map<std::string, llvm::SmallVector<mlir::hec::ComponentPortInfo, 4>>
      portDict;

  // Prepare component ports
  for (auto &op : design->getRegion(0).front())
    if (auto func = llvm::dyn_cast<mlir::tor::FuncOp>(op)) {
      llvm::SmallVector<mlir::hec::ComponentPortInfo, 4> ports;
      ports.clear();
      auto funcType = func.getType();

      size_t icount = 0;
      auto context = func.getContext();

      for (auto inPort : funcType.getInputs()) {
        auto tmpstr = std::string("in") + std::to_string(icount++);
        llvm::StringRef port_name(tmpstr);

        ports.push_back(
            hec::ComponentPortInfo(mlir::StringAttr::get(context, port_name),
                                   inPort, hec::PortDirection::INPUT));
      }

      icount += 1;
      ports.push_back(hec::ComponentPortInfo(
          mlir::StringAttr::get(context, "go"),
          mlir::IntegerType::get(context, 1), hec::PortDirection::INPUT));

      size_t ocount = 0;
      for (auto outPort : funcType.getResults()) {
        auto tmpstr = std::string("out") + std::to_string(ocount++);
        llvm::StringRef port_name(tmpstr);

        ports.push_back(
            hec::ComponentPortInfo(mlir::StringAttr::get(context, port_name),
                                   outPort, hec::PortDirection::OUTPUT));
      }

      ocount += 1;
      ports.push_back(hec::ComponentPortInfo(
          mlir::StringAttr::get(context, "done"),
          mlir::IntegerType::get(context, 1), hec::PortDirection::OUTPUT));

      portDict.insert(std::make_pair(func.getName().str(), ports));
    }

  for (auto &op : design->getRegion(0).front())
    if (auto alloc = llvm::dyn_cast<mlir::tor::AllocOp>(op)) {
      glbStorage.add_mem(alloc);
    } else if (auto constant = llvm::dyn_cast<ConstantOp>(op)) {
      glbStorage.add_constant(constant);
    }

  // Create the components
  for (auto &op : design->getRegion(0).front()) {
    if (llvm::isa<mlir::tor::FuncOp>(op)) {
      auto funcOp = llvm::dyn_cast<mlir::tor::FuncOp>(op);
      auto strategy = funcOp->getAttrOfType<StringAttr>("strategy");
      assert(strategy != nullptr);
      if (strategy.getValue() == "dynamic")
        continue;

      standardizeFunc(funcOp, rewriter);

      components.push_back(Component(funcOp, glbStorage, portDict));
    }
  }
  return 0;
}

int writeComponents(mlir::hec::DesignOp design, mlir::PatternRewriter &rewriter,
                    hecgen::GlobalStorage &glbStorage,
                    std::vector<hecgen::Component> &components) {

  for (auto component : components) {
    component.rewrite(design, rewriter);
  }
  glbStorage.codegen(design, rewriter);
  // design.dump();
  return 0;
}
} // namespace hecgen

struct HECGen : public OpRewritePattern<mlir::tor::DesignOp> {
  HECGen(MLIRContext *context) : OpRewritePattern<tor::DesignOp>(context, 1) {}

  LogicalResult matchAndRewrite(mlir::tor::DesignOp torDesign,
                                PatternRewriter &rewriter) const override {
    if (torDesign->hasAttr("staticPass")) {
      return failure();
    }

    torDesign->setAttr("staticPass", rewriter.getStringAttr("Done"));

    rewriter.setInsertionPoint(torDesign);
    /*
        auto staticPassed =
       torDesign->getAttrOfType<IntegerAttr>("staticPass");

        if (staticPassed != nullptr)
          return failure();
    */

    // torDesign->setAttr("staticPass", rewriter.getI32IntegerAttr(1));

    auto hecDesign = rewriter.create<mlir::hec::DesignOp>(torDesign.getLoc(),
                                                          torDesign.symbol());

    if (hecDesign.body().empty())
      hecDesign.body().push_back(new mlir::Block);

    std::cerr << "Create hec design" << std::endl;

    std::vector<hecgen::Component> components;

    hecgen::GlobalStorage glbStorage(hecDesign, rewriter);

    if (hecgen::genComponents(torDesign, rewriter, glbStorage, components))
      return failure();

    if (hecgen::writeComponents(hecDesign, rewriter, glbStorage, components)) {
      llvm::errs() << "FAILED\n";
      return failure();
    }

    // rewriter.eraseOp(torDesign);

    return success();
  }
};

struct HECGenPass : public HECGenBase<HECGenPass> {
  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();

    // mlir::ConversionTarget target(getContext());
    // target.addLegalDialect<mlir::hec::HECDialect>();
    // target.addLegalDialect<mlir::tor::TORDialect>();
    // target.addLegalDialect<mlir::StandardOpsDialect>();

    // target.addIllegalDialect<mlir::tor::TORDialect>();

    if (m.walk([&](tor::DesignOp design) {
           mlir::RewritePatternSet patterns(&getContext());
           patterns.insert<HECGen>(m.getContext());
           if (failed(applyOpPatternsAndFold(design, std::move(patterns))))
             return WalkResult::advance();
           return WalkResult::advance();
         }).wasInterrupted()) {
      return signalPassFailure();
    }

    // m.dump();
  }
};
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createHECGenPass() {
  return std::make_unique<HECGenPass>();
}

} // namespace mlir
