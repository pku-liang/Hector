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

#include "TOR/TORDialect.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"

#define DEBUG_TYPE "dump-scf"


namespace {
    using namespace mlir;
    using std::string;
    using nlohmann::json;
    namespace dump_scf {
        int attr_num;

        string get_opertion_attr() {
            return "op_" + std::to_string(attr_num++);
        }

        string get_type(Type type) {
            if (type.isInteger(1))
                return "bool";
            if (type.isIndex()) {
                return "i32";
            }
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
            } else {
                attr.dump();
                assert(false && "Undefined attribute");
                return "";
            }
        }

        string get_dump(Operation *op) {
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
                    return get_dump(block->getParentOp());
                } else {
                    return get_dump(block->getParentOp()) + "_" + std::to_string(arg.getArgNumber());
                }
            } else {
                assert(false);
                return "";
            }
        }

        json get_json(Operation *op);

        json get_json(scf::ForOp forOp) {
            json j;
            j["op_type"] = "for";
            j["names"] = json::array();
            j["lb"] = get_value(forOp.lowerBound());
            j["ub"] = get_value(forOp.upperBound());
            j["step"] = get_value(forOp.step());
            j["iter_name"] = get_value(forOp.getInductionVar());
            j["iter_args"] = json::array();
            j["iter_inits"] = json::array();
            j["body"] = json::array();
            for (auto val : forOp.getIterOperands()) {
                j["iter_inits"].push_back(get_value(val));
            }
            for (auto val : forOp.getRegionIterArgs()) {
                j["iter_args"].push_back(get_value(val));
            }
            for (auto &op : *(forOp.getBody())) {
                j["body"].push_back(get_json(&op));
            }
            for (auto val : forOp.getResults()) {
                j["names"].push_back(get_value(val));
            }
            return j;
        }

        json get_json(scf::IfOp ifOp) {
            json j;
            j["op_type"] = "if";
            j["condition"] = get_value(ifOp.condition());
            j["names"] = json::array();
            j["body0"] = json::array();
            j["body1"] = json::array();
            for (auto val : ifOp.getResults()) {
                j["names"].push_back(get_value(val));
            }
            for (auto &op : *(ifOp.getBody())) {
                j["body0"].push_back(get_json(&op));
            }
            for (auto &op : *(ifOp.getBody(1))) {
                j["body1"].push_back(get_json(&op));
            }
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
        return j;                                                         \
    }

        json get_json(Operation *op) {
            json j;
            if (auto nop = dyn_cast<ConstantOp>(op)) {
                j["op_type"] = "constant";
                j["name"] = get_dump(nop);
                j["operands"] = {get_attr(nop.valueAttr())};
                j["type"] = get_type(nop.getType());
            } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                j = get_json(forOp);
            } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
                j = get_json(ifOp);
            } else if (auto cmpIOp = dyn_cast<CmpIOp>(op)) {
                j["operands"] = json::array();
                j["op_type"] = string("cmp_") + stringifyCmpIPredicate(cmpIOp.predicate()).str();
                j["name"] = get_dump(cmpIOp);
                j["type"] = get_type(cmpIOp.getResult().getType());
                for (const auto &operand : cmpIOp->getOperands()) {
                    j["operands"].push_back(get_value(operand));
                }
                return j;
            } else if (auto cmpFOp = dyn_cast<CmpFOp>(op)) {
                j["operands"] = json::array();
                j["op_type"] = string("cmp_") + stringifyCmpFPredicate(cmpFOp.predicate()).str();
                j["name"] = get_dump(cmpFOp);
                j["type"] = get_type(cmpFOp.getResult().getType());
                for (const auto &operand : cmpFOp->getOperands()) {
                    j["operands"].push_back(get_value(operand));
                }
                return j;
            } else if (auto loadOp = dyn_cast<tor::LoadOp>(op)) {
                j["op_type"] = "load";
                j["name"] = get_dump(loadOp);
                assert(loadOp.getNumOperands() == 2);
                assert(loadOp->getNumResults() == 1);
                j["index"] = get_value(loadOp.getOperand(1));
                j["memory"] = get_value(loadOp.memref());
            } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
                j["op_type"] = "yield";
                j["operands"] = json::array();
                for (auto val : yieldOp->getOperands()) {
                    j["operands"].push_back(get_value(val));
                }
            } else if (auto returnOp = dyn_cast<tor::ReturnOp>(op)) {
                j["op_type"] = "return";
                j["operands"] = json::array();
                for (auto val : returnOp->getOperands()) {
                    j["operands"].push_back(get_value(val));
                }
            } else if (auto storeOp = dyn_cast<tor::StoreOp>(op)) {
                assert(storeOp.getIndices().size() == 1);
                j["op_type"] = "store";
                j["index"] = get_value(storeOp.indices()[0]);
                j["memory"] = get_value(storeOp.memref());
                j["value"] = get_value(storeOp.value());
            } else {
                OPERATION(ShiftLeftOp, "shift_left")
                OPERATION(AddIOp, "add")
                OPERATION(MulIOp, "mul")
                OPERATION(SubIOp, "sub")
                OPERATION(SIToFPOp, "sitofp")
                OPERATION(IndexCastOp, "index_cast")

                OPERATION(AddFOp, "add")
                OPERATION(MulFOp, "mul")
                OPERATION(SubFOp, "sub")
                OPERATION(TruncateIOp, "trunc")
                OPERATION(SelectOp, "select")

                op->dump();
                assert(false);
            }
            return j;
        }

        json get_json(tor::FuncOp funcOp) {
            json j;
            j["name"] = funcOp.getName();
            j["args"] = json::array();
            j["types"] = json::array();
            j["body"] = json::array();
            //TODO: types & args

//            funcOp.getType().getInputs()
            for (auto &op : *(funcOp.getBodyBlock())) {
                j["body"].push_back(get_json(&op));
            }
            return j;
        }

        json get_json(tor::DesignOp designOp) {
            json j;
            j["level"] = "software";
            j["memory"] = json::array();
            j["modules"] = json::array();

            for (auto &op : *(designOp.getBody())) {
                if (auto allocOp = dyn_cast<tor::AllocOp>(op)) {
                    json sj;
                    sj["name"] = get_dump(allocOp);
                    auto mem_type = allocOp.memref().getType().dyn_cast<tor::MemRefType>();
                    sj["size"] = mem_type.getShape()[0];
                    sj["type"] = get_type(mem_type.getElementType());
                    j["memory"].push_back(sj);
                } else if (auto funcOp = dyn_cast<tor::FuncOp>(op)) {
                    j["modules"].push_back(get_json(funcOp));
                } else {
                    op.dump();
                    assert(false);
                }
            }
            return j;
        }

        struct SCFDumpPass : SCFDumpBase<SCFDumpPass> {
            void runOnOperation() override {
                auto designOp = getOperation();
                designOp.walk([&](Operation *op) {
                    op->setAttr("dump", StringAttr::get(&getContext(), get_opertion_attr().c_str()));
                });

                designOp.walk([&](tor::DesignOp op) {
                    auto j = get_json(op);
                    std::ofstream output_file("scf.json");
                    output_file << std::setw(2) << j << std::endl;
                });

            }

        };
    }

} // namespace

namespace mlir {

    std::unique_ptr<OperationPass<tor::DesignOp>> createSCFDumpPass() {
        return std::make_unique<dump_scf::SCFDumpPass>();
    }

} // namespace mlir
