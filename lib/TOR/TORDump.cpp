#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Transforms/InliningUtils.h"

#include "TOR/TORDialect.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#include <unordered_set>
#include <string>
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"

#define DEBUG_TYPE "dump-tor"

namespace {
    using namespace mlir;
    using namespace mlir::arith;
    using std::string;
    using nlohmann::json;
    namespace dump_tor {
        int attr_num;

        string get_new_attr() {
            return "new_constant_" + std::to_string(attr_num++);
        }

        string get_tor_dump_attr() {
            return "new_tor_" + std::to_string(attr_num++);
        }

        string get_type(Type type) {
            if (type.isInteger(1))
                return "bool";
            std::string typeStr;
            llvm::raw_string_ostream stro(typeStr);
            type.print(stro);
            stro.flush();
            return typeStr;
        }

        string get_attr(Attribute attr) {
            if (auto int_attr = attr.dyn_cast<IntegerAttr>()) {
                return std::to_string(int_attr.getValue().getSExtValue());
            } else if (auto float_attr = attr.dyn_cast<FloatAttr>()) {
                double float_str = float_attr.getValue().convertToDouble();
                return std::to_string(float_str);
            } else if (auto bool_attr = attr.dyn_cast<BoolAttr>()) {
                return std::to_string(bool_attr.getValue());
            } else if (auto str_attr = attr.dyn_cast<StringAttr>()) {
                return string(str_attr.getValue());
            } else {
                attr.dump();
                assert(false && "Undefined attribute");
                return "";
            }
        }

        long long get_attr_num(Attribute attr) {
            if (auto int_attr = attr.dyn_cast<IntegerAttr>()) {
                return int_attr.getValue().getSExtValue();
            } else if (auto bool_attr = attr.dyn_cast<BoolAttr>()) {
                return bool_attr.getValue();
            } else {
                attr.dump();
                assert(false && "Undefined attribute");
                return 0;
            }
        }

        string get_dump(Operation *op) {
            if (!op->hasAttr("dump"))
                op->setAttr("dump", StringAttr::get(op->getContext(), get_tor_dump_attr().c_str()));
            return string(op->getAttr("dump").dyn_cast<StringAttr>().getValue());
        }

        string get_value(Value val) {
            if (auto op_val = val.dyn_cast<OpResult>()) {
                auto op = val.getDefiningOp();
                if (op->getNumResults() == 1) {
                    return get_dump(op);
                } else {
                    return get_dump(op) + "_" + std::to_string(op_val.getResultNumber());
                }
            } else if (auto arg = val.dyn_cast<BlockArgument>()) {
                auto block = arg.getOwner();
                if (block->getNumArguments() == 1) {
                    return get_dump(block->getParentOp()) + "_b";
                } else {
                    return get_dump(block->getParentOp()) + "_" + std::to_string(arg.getArgNumber()) + "_b";
                }
            }
            assert(false);
            return "";
        }

        json get_json(Operation *op);

        json get_json(tor::WhileOp whileOp) {
            json j;
            j["op_type"] = "while";
            j["names"] = json::array();
            j["args0"] = json::array();
            j["args1"] = json::array();
            j["iter_inits"] = json::array();
            j["body0"] = json::array();
            j["body1"] = json::array();
            for (auto arg : whileOp.getBefore().getArguments()) {
                j["args0"].push_back(get_value(arg));
            }
            for (auto arg : whileOp.getAfter().getArguments()) {
                j["args1"].push_back(get_value(arg));
            }
            for (auto val : whileOp.getInits()) {
                j["iter_inits"].push_back(get_value(val));
            }
            for (auto &op : whileOp.getBefore().front()) {
                j["body0"].push_back(get_json(&op));
            }
            for (auto &op : whileOp.getAfter().front()) {
                j["body1"].push_back(get_json(&op));
            }
            for (auto val : whileOp.getResults()) {
                j["names"].push_back(get_value(val));
            }
            j["start"] = get_attr_num(whileOp->getAttr("starttime"));
            j["end"] = get_attr_num(whileOp->getAttr("endtime"));
            return j;
        }

        json get_json(tor::ConditionOp conditionOp) {
            json j;
            j["op_type"] = "condition";
            j["names"] = json::array();
            j["condition"] = get_value(conditionOp.getCondition());
            j["args"] = json::array();
            for (auto arg : conditionOp.getArgs()) {
                j["args"].push_back(get_value(arg));
            }
            return j;
        }

        json get_json(tor::ForOp forOp) {
            json j;
            j["op_type"] = "for";
            j["names"] = json::array();
            j["lb"] = get_value(forOp.getLowerBound());
            j["ub"] = get_value(forOp.getUpperBound());
            j["step"] = get_value(forOp.getStep());
            j["iter_name"] = get_value(forOp.getInductionVar());
            j["iter_args"] = json::array();
            j["iter_inits"] = json::array();
            j["body"] = json::array();
            for (auto val: forOp.getIterOperands()) {
                j["iter_inits"].push_back(get_value(val));
            }
            for (auto val: forOp.getRegionIterArgs()) {
                j["iter_args"].push_back(get_value(val));
            }
            for (auto &op: *(forOp.getBody())) {
                j["body"].push_back(get_json(&op));
            }
            for (auto val: forOp.getResults()) {
                j["names"].push_back(get_value(val));
            }
            j["start"] = get_attr_num(forOp->getAttr("starttime"));
            j["end"] = get_attr_num(forOp->getAttr("endtime"));
            return j;
        }

        json get_json(tor::IfOp ifOp) {
            json j;
            j["op_type"] = "if";
            j["names"] = json::array();
            j["condition"] = get_value(ifOp.getCondition());
            j["body0"] = json::array();
            j["body1"] = json::array();
            if (!ifOp.getThenRegion().empty()) {
                for (auto &op: ifOp.getThenRegion().front()) {
                    j["body0"].push_back(get_json(&op));
                }
            }
            if (!ifOp.getElseRegion().empty()) {
                for (auto &op: ifOp.getElseRegion().front()) {
                    j["body1"].push_back(get_json(&op));
                }
            }
            for (auto val: ifOp.getResults()) {
                j["names"].push_back(get_value(val));
            }
            j["start"] = get_attr_num(ifOp->getAttr("starttime"));
            j["end"] = get_attr_num(ifOp->getAttr("endtime"));
            return j;
        }

#define OPERATION(TYPE, NAME)  if (auto sop = dyn_cast<TYPE>(op)) {\
        j["op_type"] = NAME;                                              \
        j["name"] = get_value(sop.getResult());                           \
        j["type"] = get_type(sop.getResult().getType());                  \
        j["operands"] = json::array();                                    \
        for (auto val : op->getOperands()) {                              \
            j["operands"].push_back(get_value(val));                      \
        }                                                                 \
        j["start"] = get_attr_num(sop->getAttr("starttime"));             \
        j["end"] = get_attr_num(sop->getAttr("endtime"));                 \
        return j;                                                         \
    }

        json get_json(Operation *op) {
            json j;
            if (auto forOp = dyn_cast<tor::ForOp>(op)) {
                j = get_json(forOp);
            } else if (auto whileOp = dyn_cast<tor::WhileOp>(op)) {
                j = get_json(whileOp);
            } else if (auto conditionOp = dyn_cast<tor::ConditionOp>(op)) {  //todo
                j = get_json(conditionOp);
            } else if (auto ifOp = dyn_cast<tor::IfOp>(op)) {
                j = get_json(ifOp);
            } else if (auto loadOp = dyn_cast<tor::LoadOp>(op)) {
                j["op_type"] = "load";
                j["name"] = get_dump(loadOp);
                assert(loadOp.getNumOperands() == 2);
                assert(loadOp->getNumResults() == 1);
                j["index"] = get_value(loadOp.getOperand(1));
                j["memory"] = get_value(loadOp.getMemref());
                j["start"] = get_attr_num(loadOp->getAttr("starttime"));
                j["end"] = get_attr_num(loadOp->getAttr("endtime"));
            } else if (auto yieldOp = dyn_cast<tor::YieldOp>(op)) {
                j["op_type"] = "yield";
                j["operands"] = json::array();
                bool hasOperands = false;
                for (auto val: yieldOp->getOperands()) {
                    j["operands"].push_back(get_value(val));
                    hasOperands = true;
                }
                j["time"] = get_attr_num(yieldOp->getParentOp()->getAttr("endtime"));
                j["jump"] = get_attr_num(yieldOp->getParentOp()->getAttr("starttime"));
                if (hasOperands) {
                    if (auto forOp = dyn_cast<tor::ForOp>(yieldOp->getParentOp())) {
                        j["operands_renew_time"] = json::array();
                        if (forOp->hasAttr("pipeline")) {
                            // pipeline yield iter args
                            for (auto [iterArg, res]: llvm::zip(forOp.getRegionIterArgs(), yieldOp->getOperands())) {
                                long long useTime = 0;
                                for (auto useOp: iterArg.getUsers()) {
                                    if (llvm::isa<tor::YieldOp>(useOp)) {
                                        continue;
                                    }
                                    useTime = std::max(useTime, get_attr_num(useOp->getAttr("endtime")) - 1);
                                }
                                auto resDefineOp = res.getDefiningOp();
                                if (resDefineOp) {
                                    useTime = std::max(useTime, get_attr_num(resDefineOp->getAttr("endtime")));
                                }
                                j["operands_renew_time"].push_back(useTime);
                            }
                        }
                    }
                }
            } else if (auto returnOp = dyn_cast<tor::ReturnOp>(op)) {
                j["op_type"] = "return";
                j["operands"] = json::array();
                for (auto val: returnOp->getOperands()) {
                    j["operands"].push_back(get_value(val));
                }
                auto funcOp = dyn_cast<tor::FuncOp>(returnOp->getParentOp());
                funcOp->walk([&](tor::TimeGraphOp op) {
                    j["time"] = op.getEndtime();
                });
            } else if (auto storeOp = dyn_cast<tor::StoreOp>(op)) {
                assert(storeOp.getIndices().size() == 1);
                j["op_type"] = "store";
                j["index"] = get_value(storeOp.getIndices()[0]);
                j["memory"] = get_value(storeOp.getMemref());
                j["value"] = get_value(storeOp.getValue());
                j["start"] = get_attr_num(storeOp->getAttr("starttime"));
                j["end"] = get_attr_num(storeOp->getAttr("endtime"));
            } else if (auto callOp = dyn_cast<tor::CallOp>(op)) {
                j["op_type"] = "call";
                j["start"] = get_attr_num(callOp->getAttr("starttime"));
                j["end"] = get_attr_num(callOp->getAttr("endtime"));
                j["names"] = json::array();
                j["operands"] = json::array();
                j["function"] = callOp.getCallee();
                for (auto return_val: callOp.getResults()) {
                    j["names"].push_back(get_value(return_val));
                }
                for (auto arg: callOp.getArgOperands()) {
                    j["operands"].push_back(get_value(arg));
                }
            } else if (auto cmpIOp = dyn_cast<tor::CmpIOp>(op)) {
                j["operands"] = json::array();
                j["start"] = get_attr_num(cmpIOp->getAttr("starttime"));
                j["end"] = get_attr_num(cmpIOp->getAttr("endtime"));
                j["op_type"] = string("cmp_") + stringifyCmpIPredicate(cmpIOp.getPredicate()).str();
                j["name"] = get_dump(cmpIOp);
                j["type"] = get_type(cmpIOp.getResult().getType());
                for (const auto &operand: cmpIOp->getOperands()) {
                    j["operands"].push_back(get_value(operand));
                }
                return j;
            } else if (auto cmpFOp = dyn_cast<tor::CmpFOp>(op)) {
                j["operands"] = json::array();
                j["start"] = get_attr_num(cmpFOp->getAttr("starttime"));
                j["end"] = get_attr_num(cmpFOp->getAttr("endtime"));
                j["op_type"] = string("cmp_") + stringifyCmpFPredicate(cmpFOp.getPredicate()).str();
                j["name"] = get_dump(cmpFOp);
                j["type"] = get_type(cmpFOp.getResult().getType());
                for (const auto &operand: cmpFOp->getOperands()) {
                    j["operands"].push_back(get_value(operand));
                }
                return j;
            } else {
                // int
                OPERATION(tor::AddIOp, "add")
                OPERATION(tor::MulIOp, "mul")
                OPERATION(tor::SubIOp, "sub")
                OPERATION(RemSIOp, "remsi")
                OPERATION(DivSIOp, "divsi")
                OPERATION(RemUIOp, "remui")
                OPERATION(DivUIOp, "divui")

                OPERATION(AndIOp, "and")
                OPERATION(OrIOp, "or")
//                OPERATION(OrIOp, "ori")
                OPERATION(XOrIOp, "xor")
                OPERATION(ShLIOp, "shift_left")
                OPERATION(ShRSIOp, "shrsi")
                OPERATION(ShRUIOp, "shrui")

                // float
                OPERATION(tor::AddFOp, "add")
                OPERATION(tor::MulFOp, "mul")
                OPERATION(tor::SubFOp, "sub")
                OPERATION(tor::DivFOp, "div")
                OPERATION(NegFOp, "negf")

                // convert
                OPERATION(SIToFPOp, "sitofp")
                OPERATION(UIToFPOp, "uitofp")
                OPERATION(FPToSIOp, "fptosi")
                OPERATION(FPToUIOp, "fptoui")
                OPERATION(ExtFOp, "extf")
                OPERATION(ExtSIOp, "extsi")
                OPERATION(ExtUIOp, "extui")
                OPERATION(TruncFOp, "truncf")
                OPERATION(TruncIOp, "trunci")
                OPERATION(IndexCastOp, "index_cast")

                // other
                OPERATION(SelectOp, "select")
//                OPERATION(TruncIOp, "trunc")

                // arithmath
                OPERATION(math::AbsFOp, "absf")
                OPERATION(math::AbsIOp, "absi")
                OPERATION(math::CeilOp, "ceil")
                OPERATION(math::FloorOp, "floor")
                OPERATION(math::RoundOp, "round")

                OPERATION(math::SqrtOp, "sqrt")
                OPERATION(math::ExpOp, "exp")
                OPERATION(math::PowFOp, "powf")
                OPERATION(math::LogOp, "log")
                OPERATION(math::ErfOp, "erf")

                // trigonometric function
                OPERATION(math::CosOp, "cos")
                OPERATION(math::TanOp, "tan")
                OPERATION(math::TanhOp, "tanh")
                OPERATION(math::SinOp, "sin")

                op->dump();
                LLVM_DEBUG(llvm::dbgs() << "[WARNING] do not support " << op->getName().getIdentifier().str() << "\n");
                assert(false);
            }
            return j;
        }

#undef OPERATION

        json get_json(tor::FuncOp funcOp) {
            json j;
            j["name"] = funcOp.getName();
            j["args"] = json::array();
            j["types"] = json::array();
            j["return_vals"] = json::array();
            j["ret_types"] = json::array();
            j["body"] = json::array();
            //TODO: types & args

            for (auto &op: funcOp.getBody().front()) {
                if (auto graph = dyn_cast<tor::TimeGraphOp>(op)) {
                    json sj;
                    sj["start"] = graph.getStarttime();
                    sj["end"] = graph.getEndtime();
                    sj["edge"] = json::array();
                    for (auto &sop: *(graph.getBody())) {
                        if (auto succOp = dyn_cast<tor::SuccTimeOp>(sop)) {
                            int to = succOp.getTime();
                            for (unsigned i = 0; i < succOp.getPoints().size(); ++i) {
                                auto from = succOp.getPoints()[i];
                                auto attrs = succOp.getEdges()[i].dyn_cast<DictionaryAttr>();
                                json edge;
                                edge["from"] = get_attr_num(from);
                                edge["to"] = to;
                                for (auto attr: attrs) {
                                    auto edge_attr = attr.getValue();
                                    if (auto str_attr = edge_attr.dyn_cast<StringAttr>()) {
                                        edge[attr.getName().strref()] = str_attr.getValue();
                                    } else {
                                        edge[attr.getName().strref()] = get_attr_num(edge_attr);
                                    }
                                }
                                sj["edge"].push_back(edge);
                            }
                        }
                    }
                    j["graph"] = sj;
                } else {
                    j["body"].push_back(get_json(&op));
                }
            }
            for (auto val: funcOp.getArguments()) {
                j["args"].push_back(get_value(val));
                j["types"].push_back(get_type(val.getType()));
            }
            for (auto val: funcOp->getResults()) {
                j["return_vals"].push_back(get_value(val));
                j["ret_types"].push_back(get_type(val.getType()));
            }

            if (funcOp->hasAttr("strategy")) {
                if (funcOp->hasAttr("pipeline")) {
                    j["strategy"] =
                            "pipeline " + get_attr(funcOp->getAttr("pipeline")) + " " + get_attr(funcOp->getAttr("II"));
                } else {
                    j["strategy"] = string(funcOp->getAttr("strategy").dyn_cast<StringAttr>().getValue());
                }
            } else {
                j["strategy"] = "static";
            }
            return j;
        }

        json get_json(tor::DesignOp designOp) {
            json j;
            j["level"] = "tor";
            j["memory"] = json::array();
            j["stream"] = json::array();
            j["m_axi"] = json::array();
            j["modules"] = json::array();
            j["constants"] = json::array();
            std::unordered_set<std::string> buses;
            for (auto &op: designOp.getBody().front()) {
                if (auto allocOp = dyn_cast<tor::AllocOp>(op)) {
                    json sj;
                    sj["name"] = get_dump(allocOp);
                    auto mem_type = allocOp.getMemref().getType().dyn_cast<tor::MemRefType>();
                    sj["size"] = mem_type.getShape()[0];
                    sj["type"] = get_type(mem_type.getElementType());
                    if (allocOp->hasAttr("value_h")) {
                        sj["value_h"] = string(allocOp->getAttr("value_h").dyn_cast<StringAttr>().getValue());
                    }
                    j["memory"].push_back(sj);
                } else if (auto funcOp = dyn_cast<tor::FuncOp>(op)) {
                    j["modules"].push_back(get_json(funcOp));
                } else if (auto nop = dyn_cast<ConstantOp>(op)) {
                    json sj;
                    sj["name"] = get_dump(nop);
                    sj["operands"] = get_attr(nop.getValueAttr());
                    sj["type"] = get_type(nop.getType());
                    j["constants"].push_back(sj);
                } else {
                    op.dump();
                    assert(false);
                }
            }

            for (auto bus: buses) {
                json sj;
                sj["name"] = bus;
                j["m_axi"].push_back(sj);
            }
            return j;
        }

        struct TORDumpPass : TORDumpBase<TORDumpPass> {
            void runOnOperation() override {
                auto designOp = getOperation();
                auto json_file = (this->json).getValue();
               designOp.walk([&](ConstantOp op) {
                    if (!op->hasAttr("dump")) {
                        op->setAttr("dump", StringAttr::get(&getContext(), get_new_attr().c_str()));
                    }
                });

                designOp.walk([&](tor::DesignOp op) {
                    auto j = get_json(op);
                    if (json_file != "") {
                        std::ofstream output_file( json_file );
                        output_file << std::setw(2) << j << std::endl;
                    }
                });

            }

        };
    }
} // namespace

namespace mlir {

    std::unique_ptr<OperationPass<tor::DesignOp>> createTORDumpPass() {
        return std::make_unique<dump_tor::TORDumpPass>();
    }

} // namespace mlir
