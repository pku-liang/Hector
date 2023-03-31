#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
    using std::string;
    using nlohmann::json;
    namespace dump_hec {
        int attr_num;

        string get_comb_attr() {
            return "comb_" + std::to_string(attr_num++);
        }

        string get_dump(Operation *op) {
            return string(op->getAttr("dump").dyn_cast<StringAttr>().getValue());
        }

        string get_attr(Attribute attr) {
            if (auto int_attr = attr.dyn_cast<IntegerAttr>()) {
                return std::to_string(int_attr.getValue().getSExtValue());
            } else if (auto float_attr = attr.dyn_cast<FloatAttr>()) {
                return float_attr.getValue().bitcastToAPInt().toString(10, false);
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
                        return string(primitive.instanceName()) + string(".") +
                               get_attr(primitive.getPrimitivePortInfo()[op_val.getResultNumber()].name);
                    } else if (auto instance = dyn_cast<hec::InstanceOp>(op)) {
                        return string(instance.instanceName()) + string(".") +
                               get_attr(instance.getReferencedComponent().portNames()[op_val.getResultNumber()]);
                    } else if (auto wire = dyn_cast<hec::WireOp>(op)) {
                        return wire.name().str();
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
                    return get_attr(component.portNames()[arg.getArgNumber()]);
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
    if (sop.guard()) {                                            \
        j["condition"] = get_value(sop.guard());                  \
    }                                                             \
    return j;                                                     \
}

        json get_json(Operation *op) {
            json j;
            if (auto cmpIOp = dyn_cast<hec::CmpIOp>(op)) {
                j["operands"] = json::array();
                j["op_type"] = string("cmp_") + cmpIOp.type().str();
                j["name"] = get_dump(cmpIOp);
                j["type"] = get_type(cmpIOp.getResult().getType());
                for (const auto &operand : cmpIOp->getOperands()) {
                    j["operands"].push_back(get_value(operand));
                }
                if (cmpIOp.guard()) {
                    j["condition"] = get_value(cmpIOp.guard());
                }
                return j;
            } else if (auto assign = dyn_cast<hec::AssignOp>(op)) {
                json j;
                j["op_type"] = "assign";
                j["src"] = get_value(assign.src());
                j["dst"] = get_value(assign.dest());
                if (assign.guard()) {
                    j["condition"] = get_value(assign.guard());
                }
                return j;
            } else if (auto enable = dyn_cast<hec::EnableOp>(op)) {
                json j;
                j["op_type"] = "enable";
                j["port"] = get_value(enable.port());
                return j;
            } else {
                OPERATION(hec::AddIOp, "add")
                OPERATION(hec::NotOp, "not")
                OPERATION(hec::ShiftLeftOp, "shift_left")
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
            for (auto &op : *state.getBody()) {
                if (auto transition = dyn_cast<hec::TransitionOp>(op)) {
                    json sj;
                    sj["jump"] = json::array();
                    for (auto &sop : *transition.getBody()) {
                        if (auto jump = dyn_cast<hec::GotoOp>(sop)) {
                            if (jump.cond()) {
                                json ssj;
                                ssj["dest"] = jump.dest();
                                ssj["cond"] = get_value(jump.cond());
                                sj["jump"].push_back(ssj);
                            } else {
                                sj["default"] = jump.dest();
                            }
                        } else if (auto done = dyn_cast<hec::DoneOp>(sop)) {
                            sj["done"] = json::array();
                            sj.erase("jump");
                            for (auto val : done->getResults()) {
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
                    sj["instance"] = go.name();
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
            for (auto &op : *stage.getBody()) {
                if (auto transition = dyn_cast<hec::TransitionOp>(op)) {
                    json sj;
                    sj["jump"] = json::array();
                    for (auto &sop : *transition.getBody()) {
                        if (auto jump = dyn_cast<hec::GotoOp>(sop)) {
                            if (jump.cond()) {
                                json ssj;
                                ssj["dest"] = jump.dest();
                                ssj["cond"] = get_value(jump.cond());
                                sj["jump"].push_back(ssj);
                            } else {
                                sj["default"] = jump.dest();
                            }
                        }
                    }
                    j["transition"] = sj;
                } else if (isa<hec::AssignOp, hec::EnableOp>(op)) {
                    j["ops"].push_back(get_json(&op));
                } else if (auto deliver = dyn_cast<hec::DeliverOp>(op)) {
                    json sj;
                    sj["op_type"] = "deliver";
                    sj["src"] = get_value(deliver.src());
                    sj["dst_port"] = get_value(deliver.destPort());
                    sj["dst_reg"] = get_value(deliver.destReg());
                    if (deliver.guard()) {
                        sj["condition"] = get_value(deliver.guard());
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
            j["num_in"] = component.numInPorts();

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
            } else if (style == "dynamic") {
                assert(false);
            }
            for (auto &op : *(component.getBody())) {
                if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
                    json sj;
                    sj["types"] = json::array();
                    sj["op_type"] = primitive.primitiveName();
                    sj["name"] = primitive.instanceName();
                    for (auto val : primitive->getResults()) {
                        sj["types"].push_back(get_type(val.getType()));
                    }
                    j["units"].push_back(sj);
                } else if (auto init = dyn_cast<hec::InitOp>(op)) {
                    json sj;
                    sj["src"] = get_value(init.src());
                    sj["dst"] = get_value(init.dst());
                    j["inits"].push_back(sj);
                } else if (auto wire = dyn_cast<hec::WireOp>(op)) {
                    json sj;
                    sj["name"] = wire.name();
                    j["wires"] = sj;
                } else if (auto states = dyn_cast<hec::StateSetOp>(op)) {
                    for (auto &sop : *(states.getBody())) {
                        auto state = dyn_cast<hec::StateOp>(sop);
                        if (state.initial()) {
                            j["init_state"] = state.getName();
                        }
                        j["states"].push_back(get_json(state));
                    }
                } else if (auto stages = dyn_cast<hec::StageSetOp>(op)) {
                    for (auto &sop : *(stages.getBody())) {
                        auto stage = dyn_cast<hec::StageOp>(sop);
                        j["stages"].push_back(get_json(stage));
                    }
                } else if (auto instance = dyn_cast<hec::InstanceOp>(op)) {
                    json sj;
                    sj["instance_name"] = instance.instanceName();
                    sj["module_name"] = instance.componentName();
                    sj["names"] = json::array();
                    j["instances"].push_back(sj);
                } else {
                    op.dump();
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

            for (auto &op : *(designOp.getBody())) {
                if (auto component = dyn_cast<hec::ComponentOp>(op)) {
                    j["modules"].push_back(get_json(component));
                } else if (auto nop = dyn_cast<ConstantOp>(op)) {
                    json sj;
                    sj["name"] = get_dump(nop);
                    sj["operands"] = get_attr(nop.valueAttr());
                    sj["type"] = get_type(nop.getType());
                    j["constants"].push_back(sj);
                } else if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
                    if (primitive.primitiveName() == "mem") {
                        json sj;
                        sj["name"] = primitive.instanceName();
                        sj["type"] = get_type(primitive->getResult(2).getType());
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
                            hec::OrOp, hec::CmpIOp>(op)) {
                        op->setAttr("dump", StringAttr::get(&getContext(), get_comb_attr().c_str()));
                    }
                });

                designOp.walk([&](hec::DesignOp op) {
                    auto j = get_json(op);
                    std::ofstream output_file("hec.json");
                    output_file << std::setw(2) << j << std::endl;
//                    std::cout << std::setw(2) << j << std::endl;
                });
                exit(-1);

            }

        };
    }

} // namespace

namespace mlir {

    std::unique_ptr<OperationPass<hec::DesignOp>> createHECDumpPass() {
        return std::make_unique<dump_hec::HECDumpPass>();
    }

} // namespace mlir
