#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"

#include "HEC/HECDialect.h"
#include "HEC/PassDetail.h"
#include "HEC/Passes.h"

#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"

#define DEBUG_TYPE "dump-hec"


namespace {
    using namespace mlir;
    using namespace mlir::arith;
    using std::string;
    using nlohmann::json;
    namespace dump_hec {
        int attr_num;

        string get_comb_attr() {
            return "comb_" + std::to_string(attr_num++);
        }

        string get_const_attr() {
            return "const_" + std::to_string(attr_num++);
        }

        string get_dump(Operation *op) {
            op->dump();
            return string(op->getAttr("dump").dyn_cast<StringAttr>().getValue());
        }

        string get_attr(Attribute attr) {
            if (auto bool_attr = attr.dyn_cast<BoolAttr>()) {
                return std::to_string(bool_attr.getValue());
            } else if (auto int_attr = attr.dyn_cast<IntegerAttr>()) {
                return std::to_string(int_attr.getValue().getSExtValue());
            } else if (auto float_attr = attr.dyn_cast<FloatAttr>()) {
                double float_str = float_attr.getValue().convertToDouble();
                return std::to_string(float_str);
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

        string get_value(Value val) {
            if (auto op_val = val.dyn_cast<OpResult>()) {
                auto op = val.getDefiningOp();
                if (op->hasAttr("dump")) {
                    if (op->getNumResults() == 1) {
                        return get_dump(op);
                    } else {
                        return get_dump(op) + "_" + std::to_string(op_val.getResultNumber());
                    }
                } else {
                    if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
                        return string(primitive.getInstanceName()) + string(".") +
                               get_attr(primitive.getPrimitivePortInfo()[op_val.getResultNumber()].name);
                    } else if (auto instance = dyn_cast<hec::InstanceOp>(op)) {
                        return string(instance.getInstanceName()) + string(".") +
                               get_attr(instance.getReferencedComponent().getPortNames()[op_val.getResultNumber()]);
                    } else if (auto wire = dyn_cast<hec::WireOp>(op)) {
                        return wire.getName().str();
                    } else {
                        op->dump();
                        assert(false);
                        return "";
                    }
                }
            } else if (auto arg = val.dyn_cast<BlockArgument>()) {
                auto block = arg.getOwner();
                auto op = block->getParentOp();
                if (auto component = dyn_cast<hec::ComponentOp>(op)) {
                    return get_attr(component.getPortNames()[arg.getArgNumber()]);
                } else {
                    op->dump();
                    assert(false);
                    return "";
                }
            } else {
                assert(false);
                return "";
            }
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

#define OPERATION(TYPE, NAME) if (auto sop = dyn_cast<TYPE>(op)) {\
    json j;                                                       \
    j["operands"] = json::array();                                \
    j["op_type"] = NAME;                                          \
    j["name"] = get_dump(sop);                                    \
    j["type"] = get_type(sop.getResult().getType());              \
    for (const auto &operand : sop->getOperands()) {              \
        j["operands"].push_back(get_value(operand));              \
    }                                                             \
    if (sop.getGuard()) {                                            \
        j["condition"] = get_value(sop.getGuard());                  \
    }                                                             \
    return j;                                                     \
}

        json get_json(Operation *op) {
            json j;
            if (auto cmpIOp = dyn_cast<hec::CmpIOp>(op)) {
                j["operands"] = json::array();
                j["op_type"] = string("cmp_") + cmpIOp.getPred().str();
                j["name"] = get_dump(cmpIOp);
                j["type"] = get_type(cmpIOp.getResult().getType());
                for (const auto &operand : cmpIOp->getOperands()) {
                    j["operands"].push_back(get_value(operand));
                }
                if (cmpIOp.getGuard()) {
                    j["condition"] = get_value(cmpIOp.getGuard());
                }
                return j;
            } else if (auto assign = dyn_cast<hec::AssignOp>(op)) {
                json j;
                j["op_type"] = "assign";
                j["src"] = get_value(assign.getSrc());
                j["dst"] = get_value(assign.getDest());
                if (assign.getGuard()) {
                    j["condition"] = get_value(assign.getGuard());
                }
                return j;
            } else if (auto enable = dyn_cast<hec::EnableOp>(op)) {
                json j;
                j["op_type"] = "enable";
                j["port"] = get_value(enable.getPort());
                return j;
            } else {
                OPERATION(hec::ShiftLeftOp, "shift_left")
                OPERATION(hec::AddIOp, "add")
                OPERATION(hec::SubIOp, "sub")
                OPERATION(hec::NotOp, "not")
                OPERATION(hec::TruncateIOp, "trunc")
                OPERATION(hec::SelectOp, "select")
                OPERATION(hec::AndOp, "and")
                OPERATION(hec::OrOp, "or")
                OPERATION(hec::XOrOp, "xor")
                op->dump();
                assert(false);
                return j;
            }
        }

#undef OPERATION

        json get_json(hec::StateOp state) {
            json j;
            j["state"] = state.getName();
            j["ops"] = json::array();
            for (auto &op : state.getBody().front()) {
                if (auto transition = dyn_cast<hec::TransitionOp>(op)) {
                    json sj;
                    sj["jump"] = json::array();
                    for (auto &sop : transition.getBody().front()) {
                        if (auto jump = dyn_cast<hec::GotoOp>(sop)) {
                            if (jump.getCond()) {
                                json ssj;
                                ssj["dest"] = jump.getDest();
                                ssj["cond"] = get_value(jump.getCond());
                                sj["jump"].push_back(ssj);
                            } else {
                                sj["default"] = jump.getDest();
                            }
                        } else if (auto done = dyn_cast<hec::DoneOp>(sop)) {
                            sj["done"] = json::array();
                            sj.erase("jump");
                            for (auto val : done->getOperands()) {
                                sj["done"].push_back(get_value(val));
                            }
                        }
                    }
                    j["transition"] = sj;
                } else if (isa<hec::AssignOp, hec::EnableOp>(op)) {
                    j["ops"].push_back(get_json(&op));
                } else if (auto go = dyn_cast<hec::GoOp>(op)) {
                    json sj;
                    sj["op_type"] = "go";
                    sj["instance"] = go.getName();
                    j["ops"].push_back(sj);
                } else {
                    j["ops"].push_back(get_json(&op));
                }
            }
            return j;
        }

        json get_json(hec::StageOp stage) {
            json j;
            j["stage"] = stage.getName();
            j["ops"] = json::array();
            for (auto &op : stage.getBody().front()) {
                if (auto transition = dyn_cast<hec::TransitionOp>(op)) {
                    json sj;
                    sj["jump"] = json::array();
                    for (auto &sop : transition.getBody().front()) {
                        if (auto jump = dyn_cast<hec::GotoOp>(sop)) {
                            if (jump.getCond()) {
                                json ssj;
                                ssj["dest"] = jump.getDest();
                                ssj["cond"] = get_value(jump.getCond());
                                sj["jump"].push_back(ssj);
                            } else {
                                sj["default"] = jump.getDest();
                            }
                        }
                    }
                    j["transition"] = sj;
                } else if (isa<hec::AssignOp, hec::EnableOp>(op)) {
                    j["ops"].push_back(get_json(&op));
                } else if (auto deliver = dyn_cast<hec::DeliverOp>(op)) {
                    json sj;
                    sj["op_type"] = "deliver";
                    sj["src"] = get_value(deliver.getSrc());
                    sj["dst_port"] = get_value(deliver.getDestPort());
                    sj["dst_reg"] = get_value(deliver.getDestReg());
                    if (deliver.getGuard()) {
                        sj["condition"] = get_value(deliver.getGuard());
                    }
                    j["ops"].push_back(sj);
                } else {
                    j["ops"].push_back(get_json(&op));
                }
            }
            return j;
        }

        json get_json(hec::ComponentOp component) {
            json j;
            j["name"] = component.getName();
            j["args"] = json::array();
            j["types"] = json::array();
            j["return_vals"] = json::array();
            j["ret_types"] = json::array();
            j["units"] = json::array();
            j["instances"] = json::array();
            j["num_in"] = component.getNumInPorts();

            auto style = get_attr(component->getAttr("style"));
            j["style"] = style;

            for (auto val : component.getArguments()) {
                j["args"].push_back(get_value(val));
                j["types"].push_back(get_type(val.getType()));
            }
            for (auto val : component->getResults()) {
                j["return_vals"].push_back(get_value(val));
                j["ret_types"].push_back(get_type(val.getType()));
            }

            if (style == "STG") {
                j["states"] = json::array();
            } else if (style == "pipeline") {
                j["pipeline_style"] = get_attr(component->getAttr("pipeline"));
                j["stages"] = json::array();
                j["inits"] = json::array();
                j["ii"] = get_attr_num(component->getAttr("II"));
            } else if (style == "handshake") {
                j["graph"] = json::array();
                j["sinks"] = json::array();
                for (auto &op : component.getBody().front()) {
                    if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
                        auto portInfo = primitive.getPrimitivePortInfo();
                        for (unsigned idx = 0; idx < portInfo.size(); ++idx) {
                            if (portInfo[idx].direction == hec::PortDirection::OUTPUT) {
                                auto port = primitive.getResult(idx);
                                int use_count = 0;
                                for (auto &bval : port.getUses()) {
                                    if (auto assign = dyn_cast<hec::AssignOp>(bval.getOwner())) {
                                        if (assign.getSrc() == port) {
                                            ++use_count;
                                        }
                                    }
                                }
                                if (use_count == 0) {
                                    j["sinks"].push_back(get_value(port));
                                }
                            }
                        }
                    }
                }
            }
            for (auto &op : component.getBody().front()) {
                if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
                    json sj;
                    sj["types"] = json::array();
                    sj["op_type"] = primitive.getPrimitiveName();
                    sj["name"] = primitive.getInstanceName();
                    for (auto val : primitive->getResults()) {
                        sj["types"].push_back(get_type(val.getType()));
                    }
                    j["units"].push_back(sj);
                } else if (auto init = dyn_cast<hec::InitOp>(op)) {
                    json sj;
                    sj["src"] = get_value(init.getSrc());
                    sj["dst"] = get_value(init.getDst());
                    j["inits"].push_back(sj);
                } else if (auto wire = dyn_cast<hec::WireOp>(op)) {
                    json sj;
                    sj["name"] = wire.getName();
                    j["wires"] = sj;
                } else if (auto states = dyn_cast<hec::StateSetOp>(op)) {
                    for (auto &sop : states.getBody().front()) {
                        auto state = dyn_cast<hec::StateOp>(sop);
                        if (state.getInitial()) {
                            j["init_state"] = state.getName();
                        }
                        j["states"].push_back(get_json(state));
                    }
                } else if (auto stages = dyn_cast<hec::StageSetOp>(op)) {
                    for (auto &sop : stages.getBody().front()) {
                        auto stage = dyn_cast<hec::StageOp>(sop);
                        j["stages"].push_back(get_json(stage));
                    }
                } else if (auto instance = dyn_cast<hec::InstanceOp>(op)) {
                    json sj;
                    sj["instance_name"] = instance.getInstanceName();
                    sj["module_name"] = instance.getComponentName();
                    sj["names"] = json::array();
                    j["instances"].push_back(sj);
                } else if (auto graph = dyn_cast<hec::GraphOp>(op)) {
                    for (auto &sop : graph.getBody().front()) {
                        j["graph"].push_back(get_json(&sop));
                    }
                } else {
                    op.dump();
                    assert(false);
                }
            }

            return j;
        }

        json get_json(hec::DesignOp designOp) {
            json j;
            j["level"] = "hec";
            j["memory"] = json::array();
            j["modules"] = json::array();
            j["constants"] = json::array();

            for (auto &op : designOp.getBody().front()) {
                if (auto component = dyn_cast<hec::ComponentOp>(op)) {
                    j["modules"].push_back(get_json(component));
                } else if (auto nop = dyn_cast<ConstantOp>(op)) {
                    json sj;
                    sj["name"] = get_dump(nop);
                    sj["operands"] = get_attr(nop.getValueAttr());
                    sj["type"] = get_type(nop.getType());
                    j["constants"].push_back(sj);
                } else if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
                    if (primitive.getPrimitiveName() == "mem") {
                        json sj;
                        sj["name"] = primitive.getInstanceName();
                        auto attr = primitive->getAttrOfType<StringAttr>("ports").getValue();
                        if (attr == "rw") {
                            sj["type"] = get_type(primitive->getResult(3).getType());
                        } else {
                            sj["type"] = get_type(primitive->getResult(2).getType());
                        }
                        sj["size"] = get_attr_num(primitive->getAttr("len"));
                        j["memory"].push_back(sj);
                    } else {
                        assert(false);
                    }
                } else {
                    op.dump();
//                    assert(false);
                }
            }
            return j;
        }

        struct HECDumpPass : HECDumpBase<HECDumpPass> {
            void runOnOperation() override {
                auto designOp = getOperation();

                designOp.walk([&](Operation *op) {
                    if (isa<hec::ShiftLeftOp, hec::AddIOp, hec::SubIOp, hec::NotOp, hec::XOrOp, hec::AndOp,
                            hec::OrOp, hec::CmpIOp, hec::TruncateIOp, hec::SelectOp>(op)) {
                        op->setAttr("dump", StringAttr::get(&getContext(), get_comb_attr().c_str()));
                    }
                });

                designOp.walk([&](ConstantOp op) {
                    if (!op->hasAttr("dump")) {
                        op->setAttr("dump", StringAttr::get(&getContext(), get_const_attr().c_str()));
                    }
                });

                designOp.walk([&](hec::DesignOp op) {
                    auto j = get_json(op);
                    std::ofstream output_file("hec.json");
                    output_file << std::setw(2) << j << std::endl;
                });
            }

        };
    }

} // namespace

namespace mlir {

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createHECDumpPass() {
        return std::make_unique<dump_hec::HECDumpPass>();
    }

} // namespace mlir
