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
            }
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
                    return get_dump(block->getParentOp());
                } else {
                    return get_dump(block->getParentOp()) + "_" + std::to_string(arg.getArgNumber());
                }
            }
        }

        string get_type(Type type) {
            std::string typeStr;
            llvm::raw_string_ostream stro(typeStr);
            type.print(stro);
            stro.flush();
            return typeStr;
        }

        json get_json(hec::ComponentOp component) {
            json j;
            j["name"] = component.getName();
            j["args"] = json::array();
            j["types"] = json::array();
            j["body"] = json::array();

            auto style = get_attr(component->getAttr("style"));
            j["style"] = style;

            for (auto val : component.getArguments()) {
                j["args"].push_back(get_value(val));
                j["types"].push_back(get_type(val.getType()));
            }

            if (style == "STG") {

            } else if (style == "pipeline") {

            } else if (style == "dynamic") {

            }
            for (auto &op : *(component.getBody())) {
                op.dump();
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
                /*if (auto allocOp = dyn_cast<tor::AllocOp>(op)) {
                    json sj;
                    sj["name"] = get_dump(allocOp);
                    auto mem_type = allocOp.memref().getType().dyn_cast<tor::MemRefType>();
                    sj["size"] = mem_type.getShape()[0];
                    sj["type"] = get_type(mem_type.getElementType());
                    j["memory"].push_back(sj);
                } else */if (auto component = dyn_cast<hec::ComponentOp>(op)) {
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
//                        sj["name"] = get_dump(primitive);
                        sj["type"] = get_type(primitive->getResult(2).getType());
                        sj["size"] = get_attr_num(primitive->getAttr("len"));
                        sj["memory"].push_back(sj);
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

                designOp.walk([&](hec::DesignOp op) {
                    auto j = get_json(op);
//                    std::ofstream output_file("hec.json");
//                    output_file << std::setw(2) << j << std::endl;
                    std::cout << std::setw(2) << j << std::endl;
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
