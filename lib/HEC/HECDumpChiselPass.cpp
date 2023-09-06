#include "HEC/PassDetail.h"
#include "mlir/Analysis/Utils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "HEC/HEC.h"
#include "HEC/HECDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include <mlir/Transforms/DialectConversion.h>
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>
#include <fstream>

#define DEBUG_TYPE "dump-chisel"

namespace mlir {
    namespace DUMP {
        using std::string;
        int var_count = 0, module_count = 0;
        std::map<detail::ValueImpl *, string> variable_names;
        std::map<string, std::vector<string>> portNames;
        std::map<string, int> numInPorts;
        std::map<Operation *, std::vector<std::pair<Value, int>>> memOutPorts;
        std::map<Operation *, std::vector<std::pair<Value, int>>> memInPorts;
        std::map<string, int> arbiterNums;

        struct memPortInfo {
            string portName, type, data, name;
            Value val;

            memPortInfo() : portName(), type(), val(), data(), name() {}

            memPortInfo(string _pn, string _t, Value _val, string _d, string _n) : portName(_pn), type(_t), val(_val),
                                                                                   data(_d), name(_n) {}

            memPortInfo operator()(string _pn) {
                memPortInfo newPort = *this;
                newPort.portName = _pn;
                return newPort;
            }
        };

        std::map<string, std::vector<memPortInfo>> memOutPortNames;
        std::map<string, std::vector<memPortInfo>> memInPortNames;

        string new_var() {
            return "var" + std::to_string(var_count++);
        }

        //        string new_module() {
        //            return "mod" + std::to_string(module_count++);
        //        }

        //        string last_module() {
        //            return "mod" + std::to_string(module_count);
        //        }
        
        void set_name(Value val, std::string name) {
            auto tmp = val.getImpl();
            variable_names[tmp] = name;
        }

        string get_name(Value val) {
            if (val.getDefiningOp()) {
                if (auto constantOp = dyn_cast<ConstantOp>(val.getDefiningOp())) {
                    string name = "";
                    //                    llvm::raw_string_ostream out(name);
                    //                    constantOp.getValue().print(out);
                    //                    constantOp.dump();
                    auto attr = constantOp.getValue();
                    if (auto int_attr = attr.dyn_cast<IntegerAttr>()) {
                        name += std::to_string(int_attr.getValue().getSExtValue()) + ".U";
                    } else if (auto float_attr = attr.dyn_cast<FloatAttr>()) {
                        //FIXME: Hex representation for float number
                        SmallVector<char> float_num;
//                        float_attr.getValue().toString(float_num);
//                        float_attr.getValue().bitcastToAPInt().toString(float_num);
//                        for (auto cr : float_num) {
//                            name += cr;
//                        }
                        name += float_attr.getValue().bitcastToAPInt().toString(10, false);
                        name += "L.U";
                    } else if (auto bool_attr = attr.dyn_cast<BoolAttr>()) {
                        name += std::to_string(bool_attr.getValue()) + ".B";
                    } else {
                        attr.dump();
                        assert(false && "Undefined attribute");
                    }
                    return name;
                }
            }
            auto tmp = val.getImpl();
            if (variable_names.find(tmp) == variable_names.end()) {
                variable_names[tmp] = new_var();
            }
            return variable_names[tmp];
        }

        string get_bool_name(Value val) {
            string name = get_name(val);
            if (val.getType().isIntOrIndex()) {
                return name + ".asBool()";
            } else
                return name;
        }

        string getType(Type type) {
            if (auto integer = type.dyn_cast<IntegerType>()) {
                if (integer.getWidth() == 1000) {
                    return "UInt(0.W)";
                }
                return "UInt(" + std::to_string(integer.getWidth()) + ".W)";
            }
            if (type.isa<Float32Type>()) {
                return "UInt(32.W)";
            }
            if (type.isa<Float64Type>()) {
                return "UInt(64.W)";
            } else {
                type.dump();
                assert(false && "Undefined Type in HEC Dialect");
                return "????";
            }
        }

        int getWidth(Type type) {
            if (auto integer = type.dyn_cast<IntegerType>()) {
                if (integer.getWidth() == 1000) {
                    return 0;
                }
                return integer.getWidth();
            }
            if (type.isa<Float32Type>()) {
                return 32;
            }
            if (type.isa<Float64Type>()) {
                return 64;
            } else {
                type.dump();
                assert(false && "Undefined Type in HEC Dialect");
                return -1;
            }
        }

        string get(const StringRef &stringref) {
            return string(stringref.data(), stringref.size());
        }

        hec::ComponentOp get_component_op(Operation *op) {
            while (!isa<hec::ComponentOp>(op)) {
                op = op->getParentOp();
            }
            return cast<hec::ComponentOp>(op);
        }

        string getName(hec::CmpIOp cmpIOp) {
            string type = get(cmpIOp.typeAttr().getValue());
            if (type == "sle") {
                return " <= ";
            } else if (type == "slt") {
                return " < ";
            } else if (type == "sgt") {
                return " > ";
            } else if (type == "sge") {
                return " >= ";
            } else if (type == "ne") {
                return " =/= ";
            } else if (type == "eq") {
                return " === ";
            } else {
                cmpIOp->dump();
                assert(false && "Unknown cmpI type");
                return "???";
            }
        }

        void
        insert_arithmetic_primitive(string &chisel_component, hec::PrimitiveOp &primitive, hec::ComponentOp &comp, bool stall = false) {
            string moduleName = get(primitive.instanceName());
            chisel_component += "\tval " + moduleName + " = Module(new ";
            //Fixme : Consider about integer width in primitive operations
            string primName = get(primitive.primitiveName());
            if (primName == "add_integer") {
                chisel_component += "AddI()";
            } else if (primName == "sub_integer") {
                chisel_component += "SubI()";
            } else if (primName == "mul_integer") {
                chisel_component += "MulI(32, 2)";
            } else if (primName == "div_integer") {
                chisel_component += "DivI()";
            } else if (primName.find("cmp_integer") != string::npos) {
                assert(false && "Undefined cmp operation");
                chisel_component += "???()";
            } else if (primName == "mem") {
                string type = get(primitive->getAttr("ports").cast<StringAttr>().getValue());
                if (type == "r") {
                    chisel_component += "ReadMem(";
                } else if (type == "w") {
                    chisel_component += "WriteMem(";
                } else if (type == "rw") {
                    chisel_component += "ReadWriteMem(";
                }
                chisel_component +=
                        std::to_string(primitive->getAttr("len").cast<IntegerAttr>().getInt()) + ",";
                auto store_type = primitive.getResult(primitive.getNumResults() - 1).getType();
                if (auto integer = store_type.dyn_cast<IntegerType>()) {
                    chisel_component += std::to_string(integer.getWidth());
                } else if (store_type.isa<Float64Type>()) {
                    chisel_component += "64";
                } else if (store_type.isa<Float32Type>()) {
                    chisel_component += "32";
                } else {
                    store_type.dump();
                    assert(false && "Unknown store type in Memory.");
                }
                chisel_component += ")";
            } else if (primName.find("cmp_float") != string::npos) {
                auto type = primitive.getResult(0).getType();
                if (type.isa<Float64Type>()) {
                    chisel_component += "CmpFBase(2, 11, 53)";
                } else if (type.isa<Float32Type>()) {
                    chisel_component += "CmpFBase(2, 8, 24)";
                }
            } else if (primName == "mul_float") {
                auto type = primitive.getResult(2).getType();
                if (type.isa<Float64Type>()) {
                    chisel_component += "MulFBase(9, 11, 53)";
                } else if (type.isa<Float32Type>()) {
                    chisel_component += "MulFBase(4, 8, 24)";
                }
            } else if (primName == "sitofp") {
                auto type = primitive.getResult(1).getType();
                if (type.isa<Float64Type>()) {
                    chisel_component += "IntToFloatBase(9, 32, 11, 53, true)";
                } else if (type.isa<Float32Type>()) {
                    chisel_component += "IntToFloatBase(4, 32, 8, 24, true)";
                }
            } else if (primName == "fptosi") {
                auto type = primitive.getResult(1).getType();
                if (type.isa<Float64Type>()) {
                    chisel_component += "FloatToIntBase(9, 32, 11, 53, true)";
                } else if (type.isa<Float32Type>()) {
                    chisel_component += "FloatToIntBase(4, 32, 8, 24, true)";
                }
            } else if (primName == "div_float") {
                auto type = primitive.getResult(2).getType();
                if (type.isa<Float64Type>()) {
                    chisel_component += "DivFBase(30, 11, 53)";
                } else if (type.isa<Float32Type>()) {
                    chisel_component += "DivFBase(20, 8, 24)";
                }
            } else if (primName == "add_float") {
                auto type = primitive.getResult(2).getType();
                if (type.isa<Float64Type>()) {
                    chisel_component += "AddSubFBase(13, 11, 53, true)";
                } else if (type.isa<Float32Type>()) {
                    chisel_component += "AddSubFBase(8, 8, 24, true)";
                }
            } else if (primName == "sub_float") {
                auto type = primitive.getResult(2).getType();
                if (type.isa<Float64Type>()) {
                    chisel_component += "AddSubFBase(13, 11, 53, false)";
                } else if (type.isa<Float32Type>()) {
                    chisel_component += "AddSubFBase(8, 8, 24, false)";
                }
            } else {
                std::cerr << get(primitive.primitiveName()) << std::endl;
                assert(false && "Unknown primitive operation");
            }
            chisel_component += ")\n";
            if (primName.find("float") != string::npos || primName == "mul_integer") {
                if (!stall) {
                    chisel_component += "\t" + moduleName + ".ce := true.B\n";
                } else {
                    chisel_component += "\t" + moduleName + ".ce := next_continue\n";
                }
            }
            if (primName.find("cmp_float") != string::npos) {
                string type = primName.substr(primName.rfind("_") + 1);
                chisel_component += "\t" + moduleName + ".opcode := ";
                if (type == "ult") {
                    chisel_component += "4.U\n";
                } else if (type == "ule") {
                    chisel_component += "5.U\n";
                } else if (type == "ugt") {
                    chisel_component += "2.U\n";
                } else if (type == "uge") {
                    chisel_component += "3.U\n";
                } else {
                    std::cerr << type << std::endl;
                    assert(false && "not found CmpF");
                }
            }
            auto portInfos = primitive.getPrimitivePortInfo();
            for (auto val : primitive.getResults()) {
                chisel_component += "\tval " + get_name(val);
                chisel_component += " = " + moduleName + ".";
                chisel_component += get(portInfos[val.getResultNumber()].name.getValue()) + "\n";
                if (portInfos[val.getResultNumber()].direction == hec::PortDirection::INPUT) {
                    chisel_component += "\t" + get_name(val) + " := DontCare\n";
                }
            }
        }

#define insert_bitwise_operation(chisel, TYPE, OPTYPE) \
            if (auto orOp = dyn_cast<TYPE>(op)) {\
                if (orOp.guard() == Value()) {\
                    chisel += tab + "val " + get_name(orOp.res());\
                    chisel += " = " + get_name(orOp.lhs());\
                    chisel += OPTYPE + get_name(orOp.rhs()) + "\n";\
                } else {\
                    chisel += tab + "val " + get_name(orOp.res());\
                    chisel += " = Wire(" + getType(orOp.res().getType()) + ")\n";\
                    chisel += tab + get_name(orOp.res()) + " := DontCare\n";\
                    chisel += tab + "when (" + get_bool_name(orOp.guard()) + ") {\n";\
                    chisel += tab + "\t" + get_name(orOp.res());\
                    chisel += " := " + get_name(orOp.lhs());\
                    chisel += OPTYPE + get_name(orOp.rhs()) + "\n";\
                    chisel += tab + "}\n";\
                }\
            }

        void insert_arithmetic_op(string &chisel_code, string &tab, Operation *op) {
            if (auto constant = dyn_cast<ConstantOp>(op)) {
            } else if (auto cmpIOp = dyn_cast<hec::CmpIOp>(op)) {
                if (cmpIOp.guard() == Value()) {
                    chisel_code += tab + "val " + get_name(cmpIOp.res());
                    chisel_code += " = " + get_name(cmpIOp.lhs());
                    chisel_code += getName(cmpIOp) + get_name(cmpIOp.rhs()) + "\n";
                } else {
                    chisel_code += tab + "val " + get_name(cmpIOp.res());
                    chisel_code += " = Wire(" + getType(cmpIOp.res().getType()) + ")\n";
                    chisel_code += tab + get_name(cmpIOp.res()) + " := DontCare\n";
                    chisel_code += tab + "when (" + get_bool_name(cmpIOp.guard()) + ") {\n";
                    chisel_code += tab + "\t" + get_name(cmpIOp.res());
                    chisel_code += " := " + get_name(cmpIOp.lhs());
                    chisel_code += getName(cmpIOp) + get_name(cmpIOp.rhs()) + "\n";
                    chisel_code += tab + "}\n";
                }
            } else if (auto selectOp = dyn_cast<hec::SelectOp>(op)) {
                if (selectOp.guard() == Value()) {
                    chisel_code += tab + "val " + get_name(selectOp.getResult());
                    chisel_code += " = Mux(" + get_bool_name(selectOp.getOperand(0));
                    chisel_code += ", " + get_name(selectOp.getOperand(1));
                    chisel_code += ", " + get_name(selectOp.getOperand(2)) + ")\n";
                } else {
                    chisel_code += tab + "val " + get_name(selectOp.getResult());
                    chisel_code += " = Wire(" + getType(selectOp.getResult().getType()) + ")\n";
                    chisel_code += tab + get_name(selectOp.getResult()) + " := DontCare\n";
                    chisel_code += tab + "when (" + get_bool_name(selectOp.guard()) + ") {\n";
                    chisel_code += tab + "\t" + get_name(selectOp.res());
                    chisel_code += " := Mux(" + get_bool_name(selectOp.getOperand(0));
                    chisel_code += ", " + get_name(selectOp.getOperand(1));
                    chisel_code += ", " + get_name(selectOp.getOperand(2)) + ")\n";
                    chisel_code += tab + "}\n";
                }
            } else if (isa<hec::OrOp>(op)) {
                insert_bitwise_operation(chisel_code, hec::OrOp, " | ")
            } else if (isa<hec::XOrOp>(op)) {
                insert_bitwise_operation(chisel_code, hec::XOrOp, " ^ ")
            } else if (isa<hec::ShiftLeftOp>(op)) {
                insert_bitwise_operation(chisel_code, hec::ShiftLeftOp, " << ")
            } else if (isa<hec::SignedShiftRightOp>(op)) {
                insert_bitwise_operation(chisel_code, hec::SignedShiftRightOp, " >> ")
            } else if (isa<hec::AndOp>(op)) {
                insert_bitwise_operation(chisel_code, hec::AndOp, " & ")
            } else if (auto addIOp = dyn_cast<hec::AddIOp>(op)) {
                insert_bitwise_operation(chisel_code, hec::AddIOp, " + ")
            } else if (auto addIOp = dyn_cast<hec::SubIOp>(op)) {
                insert_bitwise_operation(chisel_code, hec::SubIOp, " - ")
            } else if (auto notOp = dyn_cast<hec::NotOp>(op)) {
                if (notOp.guard() == Value()) {
                    chisel_code += tab + "val " + get_name(notOp.res());
                    chisel_code += " = !" + get_name(notOp.src()) + "\n";
                } else {
                    chisel_code += tab + "val " + get_name(notOp.res());
                    chisel_code += " = Wire(" + getType(notOp.res().getType()) + ")\n";
                    chisel_code += tab + get_name(notOp.res()) + " := DontCare\n";
                    chisel_code += tab + "when (" + get_bool_name(notOp.guard()) + ") {\n";
                    chisel_code += tab + "\t" + get_name(notOp.res());
                    chisel_code += " := !" + get_name(notOp.src()) + "\n";
                    chisel_code += tab + "}\n";
                }
            } else if (auto truncIOp = dyn_cast<hec::TruncateIOp>(op)) {
                if (truncIOp.guard() == Value()) {
                    chisel_code += tab + "val " + get_name(truncIOp.res());
                    chisel_code += " = " + get_name(truncIOp.lhs()) + "\n";
                } else {
                    chisel_code += tab + "val " + get_name(truncIOp.res());
                    chisel_code += " = Wire(" + getType(truncIOp.res().getType()) + ")\n";
                    chisel_code += tab + get_name(truncIOp.res()) + " := DontCare\n";
                    chisel_code += tab + "when (" + get_bool_name(truncIOp.guard()) + ") {\n";
                    chisel_code += tab + "\t" + get_name(truncIOp.res());
                    chisel_code += " := " + get_name(truncIOp.lhs()) + "\n";
                    chisel_code += tab + "}\n";
                }
            } else if (auto negFOp = dyn_cast<hec::NegFOp>(op)) {
                auto type = negFOp.getResult().getType();
                if (negFOp.guard() == Value()) {
                    chisel_code += tab + "val " + get_name(negFOp.res());
                    if (type.isa<Float32Type>()) {
                        chisel_code += " = NegF(8, 24, " + get_name(negFOp.lhs()) + ")\n";
                    } else {
                        chisel_code += " = NegF(11, 53, " + get_name(negFOp.lhs()) + ")\n";
                    }
                } else {
                    chisel_code += tab + "val " + get_name(negFOp.res());
                    chisel_code += " = Wire(" + getType(negFOp.res().getType()) + ")\n";
                    chisel_code += tab + get_name(negFOp.res()) + " := DontCare\n";
                    chisel_code += tab + "when (" + get_bool_name(negFOp.guard()) + ") {\n";
                    chisel_code += tab + "\t" + get_name(negFOp.res());
                    if (type.isa<Float32Type>()) {
                        chisel_code += " := NegF(8, 24, " + get_name(negFOp.lhs()) + ")\n";
                    } else {
                        chisel_code += " := NegF(11, 53, " + get_name(negFOp.lhs()) + ")\n";
                    }
                    chisel_code += tab + "}\n";
                }
            } else if (auto sextIOp = dyn_cast<hec::SignExtendIOp>(op)) {
                //FIXME: Deal with sexti operation
                if (sextIOp.guard() == Value()) {
                    chisel_code += tab + "val " + get_name(sextIOp.res());
                    chisel_code += " = " + get_name(sextIOp.lhs()) + "\n";
                } else {
                    chisel_code += tab + "val " + get_name(sextIOp.res());
                    chisel_code += " = Wire(" + getType(sextIOp.res().getType()) + ")\n";
                    chisel_code += tab + get_name(sextIOp.res()) + " := DontCare\n";
                    chisel_code += tab + "when (" + get_bool_name(sextIOp.guard()) + ") {\n";
                    chisel_code += tab + "\t" + get_name(sextIOp.res());
                    chisel_code += " := " + get_name(sextIOp.lhs()) + "\n";
                    chisel_code += tab + "}\n";
                }
            } else {
                op->dump();
                assert(false && "Undefined operation in state or stage");
            }
        }


        string dumpState(hec::StateSetOp &hec, bool wrapped = false) {
            string chisel_state = "";
            chisel_state += "\tobject State extends ChiselEnum {\n\t\tval ";
            bool first = true;
            for (auto &state : *(hec.getBody())) {
                if (auto stateOp = dyn_cast<hec::StateOp>(state)) {
                    chisel_state += (first ? "" : ", ") + get(stateOp.getName());
                    first = false;
                }
            }
            chisel_state += " = Value\n\t}\n";

            for (auto &state : *(hec.getBody())) {
                if (auto stateOp = dyn_cast<hec::StateOp>(state)) {
//                    stateOp.dump();
                    if (stateOp.initial()) {
                        chisel_state += "\tval state = RegInit(State." + get(stateOp.getName()) + ")\n";
                        chisel_state += "\tval ready = go & (state === State." + get(stateOp.getName()) + ")\n";
                    }
                }
            }

            chisel_state += "\tswitch (state) {\n";
/*            for (auto &state : *(hec.getBody())) {
                if (auto stateOp = dyn_cast<hec::StateOp>(state)) {
                    if (stateOp.initial()) {
                        chisel_state += "\t\tis (State.idle) {\n";
                        chisel_state += "\t\t\twhen (go) {\n";
                        chisel_state += "\t\t\t\tstate := State." + get(stateOp.getName()) + "\n";
                        chisel_state += "\t\t\t}\n\t\t}\n";
                    }
                }
            }*/

            auto comp = cast<hec::ComponentOp>(hec->getParentOp());
            for (auto &state : *(hec.getBody())) {
                if (auto stateOp = dyn_cast<hec::StateOp>(state)) {
                    chisel_state += "\t\tis (State." + get(stateOp.getName()) + ") {\n";
                    string tab = "\t\t\t";
                    if (stateOp.initial()) {
                        chisel_state += "\t\t\twhen (go) {\n";
                        tab += "\t";
                    }
                    for (auto &op : *(stateOp.getBody())) {
                        if (auto transition = dyn_cast<hec::TransitionOp>(op)) {
                            for (auto &sop : *(transition.getBody())) {
                                if (auto branch = dyn_cast<hec::GotoOp>(sop)) {
                                    if (branch.cond() == Value()) {
                                        chisel_state += tab + "state := State." + get(branch.dest()) + ";\n";
                                    } else {
                                        chisel_state += tab + "when (" + get_bool_name(branch.cond());
                                        chisel_state +=
                                                ") {\n\t" + tab + "state := State." + get(branch.dest()) + ";\n" + tab +
                                                "}\n";
                                    }
                                }
                                if (auto done = dyn_cast<hec::DoneOp>(sop)) {
                                    if (wrapped) {
                                        chisel_state += tab + "done()\n";
                                    } else {
                                        chisel_state += tab + "done := 1.U\n";
                                    }
                                    for (unsigned idx = 0; idx < done.getNumOperands(); ++idx) {
                                        chisel_state += tab + get_name(comp.getArgument(comp.numInPorts() + idx));
                                        chisel_state += " := " + get_name(done.getOperand(idx)) + "\n";
                                    }
                                }
                                if (auto done = dyn_cast<hec::CDoneOp>(sop)) {
                                    chisel_state += tab + "when (" + get_bool_name(done.cond()) + ") {\n";
                                    if (wrapped) {
                                        chisel_state += tab + "\tdone()\n";
                                    } else {
                                        chisel_state += tab + "\tdone := 1.U\n";
                                    }
                                    for (unsigned idx = 1; idx < done.getNumOperands(); ++idx) {
                                        chisel_state +=
                                                tab + "\t" + get_name(comp.getArgument(comp.numInPorts() + idx - 1));
                                        chisel_state += " := " + get_name(done.getOperand(idx)) + "\n";
                                    }
                                    chisel_state += tab + "}\n";
                                }
                            }
                        } else if (auto goOp = dyn_cast<hec::GoOp>(op)) {
                            if (goOp.cond() == Value()) {
                                chisel_state += tab + get(goOp.name()) + ".go := 1.U\n";
                            } else {
                                chisel_state += tab + "when (" + get_bool_name(goOp.cond()) + ") {\n";
                                chisel_state += tab + "\t" + get(goOp.name()) + ".go := 1.U\n" + tab + "}\n";
                            }
                        } else if (auto assign = dyn_cast<hec::AssignOp>(op)) {
                            if (assign.guard() == Value()) {
                                chisel_state += tab + get_name(assign.dest()) + " := " + get_name(assign.src()) + "\n";
                            } else {
                                chisel_state += tab + "when (" + get_bool_name(assign.guard()) + ") {\n";
                                chisel_state +=
                                        tab + "\t" + get_name(assign.dest()) + " := " + get_name(assign.src()) + "\n";
                                chisel_state += tab + "}\n";
                            }
                        } else if (auto enableOp = dyn_cast<hec::EnableOp>(op)) {
                            if (enableOp.cond() == Value()) {
                                chisel_state += tab + get_name(enableOp.port()) + " := true.B\n";
                            } else {
                                chisel_state += tab + "when (" + get_bool_name(enableOp.cond()) + ") {\n";
                                chisel_state += tab + "\t" + get_name(enableOp.port()) + " := 1.U\n";
                                chisel_state += tab + "}\n";
                            }
                        } else {
                            insert_arithmetic_op(chisel_state, tab, &op);
                        }
                    }
                    if (stateOp.initial()) {
                        chisel_state += "\t\t\t}\n";
                    }
                    chisel_state += "\t\t}\n";
                } else {
                    state.dump();
                    assert(false && "Undefined operation in stateset");
                }
            }

            chisel_state += "\t}\n";

            return chisel_state;
        }

        string dumpStage(hec::StageSetOp &stageSet, int II, int latency, bool function = false) {
            string chisel_stage = "";

            auto comp = cast<hec::ComponentOp>(stageSet->getParentOp());

            int shift_size = ceil((latency + 1) * 1.0 / II);
            int stage_size = stageSet.getBody()->getOperations().size();

            chisel_stage += "\tval shift_register = RegInit(0.U(" + std::to_string(shift_size) + ".W))\n";
            if (!function) {
                //FIXME: last signal
                chisel_stage += "\twhen (go) {\n"
                                "\t\twhen (" + get_name(comp.getArgument(0)) + " > " + get_name(comp.getArgument(1)) +
                                ") {\n";
                /*for (unsigned idx = 0; 2 * (idx + 1) + 3 < comp.getNumArguments(); ++idx) {
                    chisel_stage += "\t\t\t" + get_name(comp.getArgument(comp.numInPorts() + idx)) + " := " +
                                    get_name(comp.getArgument(idx + 3)) + "\n";
                }*/
                chisel_stage += "\t\t\tdone := true.B\n";
                chisel_stage += "\t\t} .otherwise {\n"
                                "\t\t\tstart := true.B\n"
                                "\t\t}\n"
                                "\t}\n";
            }
            chisel_stage += "\tdef valid(stage: Int): Bool = {\n\t\t";
            for (int stage = 1; stage < stage_size; ++stage) {
                if (stage != 1) {
                    chisel_stage += " else ";
                }
                chisel_stage += "if (stage == " + std::to_string(stage) + ") {\n";
                chisel_stage += "\t\t\tshift_register(" + std::to_string((stage - 1) / II) + ")\n";
                chisel_stage += "\t\t}";
            }
            chisel_stage += " else {\n"
                            "\t\t\tnew_input\n"
                            "\t\t}\n"
                            "\t}\n";
            auto dumpOperation = [&](string &tab, Operation *op, int stage) {
                if (auto deliver = dyn_cast<hec::DeliverOp>(op)) {
                    auto done_signal = comp.getArgument(comp.getNumArguments() - 1);
                    bool isReg = deliver.destReg().getDefiningOp();
                    if (isReg) {
                        if (auto primitive = dyn_cast<hec::PrimitiveOp>(deliver.destReg().getDefiningOp())) {
                            if (primitive.primitiveName() == "register") {
                                isReg = true;
                            } else isReg = false;
                        } else isReg = false;
                    }
                    if (deliver.guard() == Value()) {
                        if (isReg) {
                            chisel_stage += tab + "when (valid(" + std::to_string(stage) + ")) {\n";
                            chisel_stage +=
                                    tab + "\t" + get_name(deliver.destReg()) + " := " + get_name(deliver.src()) +
                                    "\n";
                            if (done_signal != deliver.destPort()) {
                                chisel_stage +=
                                        tab + "\t" + get_name(deliver.destPort()) + " := " + get_name(deliver.src()) +
                                        "\n";
                            }
                            chisel_stage += tab + "}\n";
                        } else {
                            chisel_stage +=
                                    tab + get_name(deliver.destReg()) + " := " + get_name(deliver.src()) + "\n";
                            if (done_signal != deliver.destPort()) {
                                chisel_stage +=
                                        tab + get_name(deliver.destPort()) + " := " + get_name(deliver.src()) + "\n";
                            }
                        }
                    } else {
                        chisel_stage +=
                                tab + "when (" + (isReg ? "valid(" + std::to_string(stage) + ") && " : "") +
                                get_bool_name(deliver.guard()) + ") {\n";
                        chisel_stage +=
                                tab + "\t" + get_name(deliver.destReg()) + " := " + get_name(deliver.src()) +
                                "\n";
                        if (done_signal != deliver.destPort()) {
                            chisel_stage +=
                                    tab + "\t" + get_name(deliver.destPort()) + " := " + get_name(deliver.src()) + "\n";
                        }
                    }
                } else if (auto yieldOp = dyn_cast<hec::YieldOp>(op)) {
                    chisel_stage += tab + "new_output := valid(" + std::to_string(latency) + ")\n";
                    for (unsigned idx = 0; idx < yieldOp.getNumOperands(); ++idx) {
                        chisel_stage += tab + get_name(comp.getArgument(comp.numInPorts() + idx));
                        chisel_stage += " := " + get_name(yieldOp.getOperand(idx)) + "\n";
                    }
                } else if (auto enableOp = dyn_cast<hec::EnableOp>(op)) {
                    if (enableOp.cond() == Value()) {
                        chisel_stage += tab + "when (valid(" + std::to_string(stage) + ")) {\n";
                        chisel_stage += tab + "\t" + get_name(enableOp.port()) + " := 1.U\n";
                        chisel_stage += tab + "}\n";
                    } else {
                        chisel_stage +=
                                tab + "when (valid(" + std::to_string(stage) + ") && " +
                                get_bool_name(enableOp.cond()) +
                                ") {\n";
                        chisel_stage += tab + "\t" + get_name(enableOp.port()) + " := 1.U\n";
                        chisel_stage += tab + "}\n";
                    }
                } else {
                    insert_arithmetic_op(chisel_stage, tab, op);
                }
            };
            if (II != 1) {
                chisel_stage += "\tval counter = RegInit(0.U(" + std::to_string(int(ceil(log2(II)))) + ".W))\n";
                if (function) {
                    chisel_stage += "when (continue) {\n";
                }
                if (function) {
                    chisel_stage += "\twhen (new_input || shift_register =/= 0.U) {\n"
                                    "\t\twhen (counter === " + std::to_string(II - 1) +
                                    ".U) {\n"
                                    "\t\t\tcounter := 0.U\n"
                                    "\t\t}.otherwise {\n"
                                    "\t\t\tcounter := counter + 1.U\n"
                                    "\t\t}\n"
                                    "\t}.otherwise {\n"
                                    "\t\tcounter := 0.U\n"
                                    "\t}\n";
                    chisel_stage += "\tnew_input := all_valid & (counter === 0.U)\n";
                } else {
                    chisel_stage += "\twhen (counter === " + std::to_string(II - 1) +
                                    ".U) {\n"
                                    "\t\tcounter := 0.U\n"
                                    "\t}.otherwise {\n"
                                    "\t\twhen (start || shift_register =/= 0.U) {\n"
                                    "\t\t\tcounter := counter + 1.U\n"
                                    "\t\t}\n"
                                    "\t}\n";
                }
                chisel_stage += "\twhen (counter === 0.U) {\n"
                                "\t\tshift_register := Cat(shift_register(" + std::to_string(shift_size - 2) +
                                ", 0), new_input)\n\t}\n";
                for (int idx = 0; idx < II; ++idx) {
                    chisel_stage += "\twhen (counter === " + std::to_string(idx) + ".U) {\n";
                    for (int stage_idx = idx; stage_idx < stage_size; stage_idx += II) {
                        auto stage = stageSet.getBody()->begin();
                        for (unsigned next = 0; next < stage_idx; ++next) {
                            ++stage;
                        }
                        if (auto stageOp = dyn_cast<hec::StageOp>(stage)) {
                            string tab = "\t\t";
                            for (auto &op : *(stageOp.getBody())) {
                                if (auto assign = dyn_cast<hec::AssignOp>(op)) {
                                    bool isReg = assign.dest().getDefiningOp();
                                    if (isReg) {
                                        if (auto primitive = dyn_cast<hec::PrimitiveOp>(
                                                assign.dest().getDefiningOp())) {
                                            if (primitive.primitiveName() == "register") {
                                                isReg = true;
                                            } else isReg = false;
                                        } else isReg = false;
                                    }
                                    if (assign.guard() == Value()) {
                                        if (isReg) {
                                            chisel_stage += tab + "when (valid(" + std::to_string(stage_idx) + ")) {\n";
                                            chisel_stage +=
                                                    tab + "\t" + get_name(assign.dest()) + " := " +
                                                    get_name(assign.src()) +
                                                    "\n";
                                            chisel_stage += tab + "}\n";
                                        } else {
                                            chisel_stage +=
                                                    tab + get_name(assign.dest()) + " := " + get_name(assign.src()) +
                                                    "\n";
                                        }
                                    } else {
                                        chisel_stage += tab + "when (" +
                                                        (isReg ? ("valid(" + std::to_string(stage_idx) + ") && ")
                                                               : "") +
                                                        get_bool_name(assign.guard()) + ") {\n";
                                        chisel_stage +=
                                                tab + "\t" + get_name(assign.dest()) + " := " + get_name(assign.src()) +
                                                "\n";
                                        chisel_stage += tab + "}\n";
                                    }
                                } else dumpOperation(tab, &op, stage_idx);
                            }
                        } else {
                            stage->dump();
                            assert(false && "Undefined operation in stageset");
                        }
                    }
                    chisel_stage += "\t}\n";
                }
                if (function) {
                    chisel_stage += "\twhen (counter === " + std::to_string((stage_size - 1) % II) +
                                    ".U && valid(" + std::to_string(latency) + ")) {\n" +
                                    "\t\tdone := true.B\n\t}\n";
                } else {
                    chisel_stage += "\twhen (counter === " + std::to_string((stage_size - 1) % II) +
                                    ".U && valid(" + std::to_string(latency) + ")) {\n" + "\t\tdone := !valid(" +
                                    std::to_string(latency - II) + ")\n\t}\n";
                }
                if (function) {
                    chisel_stage += "}\n";
                }

            } else {
//                chisel_stage += "\twhen (start) {\n"
//                                "\t\tshift_register := Cat(shift_register(" + std::to_string(shift_size - 2) +
//                                ", 0), new_input)\n\t}\n";
                if (function) {
                    chisel_stage += "when (continue) {\n";
                }
                chisel_stage += "\tshift_register := Cat(shift_register(" + std::to_string(shift_size - 2) +
                                ", 0), new_input)\n";
                chisel_stage += "\twhen (true.B) {\n";
                int stage_idx = 0;
                for (auto &stage : *stageSet.getBody()) {
                    if (auto stageOp = dyn_cast<hec::StageOp>(stage)) {
                        string tab = "\t\t";
                        for (auto &op : *(stageOp.getBody())) {
                            if (auto assign = dyn_cast<hec::AssignOp>(op)) {
                                bool isReg = assign.dest().getDefiningOp();
                                if (isReg) {
                                    if (auto primitive = dyn_cast<hec::PrimitiveOp>(assign.dest().getDefiningOp())) {
                                        if (primitive.primitiveName() == "register") {
                                            isReg = true;
                                        } else isReg = false;
                                    } else isReg = false;
                                }
                                if (assign.guard() == Value()) {
                                    if (isReg) {
                                        chisel_stage += tab + "when (valid(" + std::to_string(stage_idx) + ")) {\n";
                                        chisel_stage +=
                                                tab + "\t" + get_name(assign.dest()) + " := " + get_name(assign.src()) +
                                                "\n";
                                        chisel_stage += tab + "}\n";
                                    } else {
                                        chisel_stage +=
                                                tab + get_name(assign.dest()) + " := " + get_name(assign.src()) + "\n";
                                    }
                                } else {
                                    chisel_stage += tab + "when (" +
                                                    (isReg ? ("valid(" + std::to_string(stage_idx) + ") && ") : "") +
                                                    get_bool_name(assign.guard()) + ") {\n";
                                    chisel_stage +=
                                            tab + "\t" + get_name(assign.dest()) + " := " + get_name(assign.src()) +
                                            "\n";
                                    chisel_stage += tab + "}\n";
                                }
                            } else dumpOperation(tab, &op, stage_idx);
                        }
                        ++stage_idx;
                    } else {
                        stage.dump();
                        assert(false && "Undefined operation in stageset");
                    }
                }
                if (function) {
                    chisel_stage +=
                            "\twhen (valid(" + std::to_string(latency) + ")) {\n" + "\t\tdone := true.B\n\t}\n";
                } else {
                    chisel_stage += "\twhen (valid(" + std::to_string(latency) + ")) {\n" + "\t\tdone := !valid(" +
                                    std::to_string(latency - II) + ")\n\t}\n";
                }
                chisel_stage += "\t}\n";
                if (function) {
                    chisel_stage += "}\n";
                }
            }
            if (!function) {
                chisel_stage += "\twhen (done) {\n"
                                "\t\tshift_register := 0.U\n";
                if (II != 1) {
                    chisel_stage += "\t\tcounter := 0.U\n";
                }
                chisel_stage += "\t}\n";
            }


            return chisel_stage;

        }

        string dumpInstance(hec::InstanceOp &instance, hec::ComponentOp &comp) {
            bool wrapped = comp.interfc() == "wrapped";
            bool sub_wrapped = false;
            string modName = get(instance.instanceName());
            string compName = get(instance.componentName());
            string chisel_instance = "\tval " + modName;
            chisel_instance += " = Module(new " + compName + ")\n";
            if (!wrapped) {
                chisel_instance += "\t" + modName + ".go := 0.U\n";
            }
            // FIXME: dummy representation
            if (wrapped) {
                auto hecDesign = cast<hec::DesignOp>(comp->getParentOp());
                hecDesign.walk([&](hec::ComponentOp op) {
                    if (op.getName() == compName) {
                        sub_wrapped = op.interfc() == "wrapped";
                        }
                });
            }
            for (unsigned idx = 0; idx < instance.getNumResults(); ++idx) {
                if (portNames[compName][idx] == "go") {
                    continue;
                }
                if (wrapped && portNames[compName][idx] == "done") {
                    continue;
                }
                if (!wrapped || sub_wrapped) {
                    set_name(instance.getResult(idx), modName + "." + portNames[compName][idx]);
                } else {
                    chisel_instance += "\tval " + get_name(instance.getResult(idx)) + " = ";
                    chisel_instance += modName + ".d_" + portNames[compName][idx] + "\n";
                }
                if (!(wrapped && sub_wrapped)) {
                    if (wrapped || idx < static_cast<unsigned int>(numInPorts[compName])) {
                        chisel_instance += "\t" + get_name(instance.getResult(idx)) + " := DontCare\n";
                    }
                }
                if (wrapped && idx >= static_cast<unsigned int>(numInPorts[compName])) {
                    if (instance.getResult(idx).use_empty()) {
                        chisel_instance += "\t" + get_name(instance.getResult(idx)) + ".ready := true.B\n";
                    }
                }
            }
            if (!wrapped) {
                auto &memOutVector = memOutPortNames[compName];
                auto &selfMemOutVector = memOutPortNames[get(comp.getName())];
                for (auto memPort : memOutVector) {
                    string name = modName + "_" + memPort.portName;
                    bool found = false;
                    for (auto const &meet : selfMemOutVector) {
                        if (meet.portName == name) {
                            found = true;
                            break;
                        }
                    }
                    if (found) {
                        continue;
                    }
                    chisel_instance += "\tval " + name + " = IO(Output(" + memPort.type + "))\n";
                    chisel_instance += "\t" + name + " := " + modName + "." + memPort.portName + "\n";
                    selfMemOutVector.push_back(memPort(name));
                }
                auto &memInVector = memInPortNames[compName];
                auto &selfMemInVector = memInPortNames[get(comp.getName())];
                for (auto memPort : memInVector) {
                    string name = modName + "_" + memPort.portName;
                    bool found = false;
                    for (auto const &meet : selfMemInVector) {
                        if (meet.portName == name) {
                            found = true;
                            break;
                        }
                    }
                    if (found) {
                        continue;
                    }
                    chisel_instance += "\tval " + name + " = IO(Input(" + memPort.type + "))\n";
                    chisel_instance += "\t" + modName + "." + memPort.portName + " := " + name + "\n";
                    selfMemInVector.push_back(memPort(name));
                }
            }
//            std::cerr << chisel_instance << std::endl;
//            exit(-1);

            return chisel_instance;
        }

        string dumpGraph(hec::GraphOp &graph) {
            string chisel_graph = "";
            for (auto &op : *(graph.getBody())) {
                if (auto assign = dyn_cast<hec::AssignOp>(op)) {
                    if (assign.src().getDefiningOp() && isa<ConstantOp>(assign.src().getDefiningOp())) {
                        chisel_graph += "\t" + get_name(assign.dest()) + ".bits := " + get_name(assign.src()) + "\n";
                        chisel_graph += "\t" + get_name(assign.dest()) + ".valid := true.B\n";
                    } else {
                        chisel_graph += "\t" + get_name(assign.dest()) + " <> " + get_name(assign.src()) + "\n";
                    }
                }
            }
            return chisel_graph;
        }

        string generateMemPorts(hec::ComponentOp &comp) {
            string chisel_mem = "";
            auto &memOutVector = memOutPorts[comp];
            auto &subOutVector = memOutPortNames[get(comp.getName())];
            for (auto memValPair : memOutVector) {
                auto memVal = memValPair.first;
                string valName = get_name(memVal);
                bool found = false;
                for (auto const &meet : subOutVector) {
                    if (meet.portName == valName) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    continue;
                }
                chisel_mem += "\tval " + valName + " = IO(Output(" + getType(memVal.getType()) + "))\n";
                auto type = memVal.getType();
                int resNum = memVal.cast<OpResult>().getResultNumber();
                if (memValPair.second == 1) {
                    if (resNum == 0) {
                        chisel_mem += "\t" + valName + " := false.B\n";
                    } else {
                        chisel_mem += "\t" + valName + " := DontCare\n";
                    }
                    if (resNum == 2) {
                        subOutVector.push_back(
                                memPortInfo(valName, getType(memVal.getType()), memVal, "bits", "w_dataIn"));
                    } else if (resNum == 0) {
                        subOutVector.push_back(
                                memPortInfo(valName, getType(memVal.getType()), memVal, "valid", "dataIn"));
                    } else if (resNum == 1) {
                        subOutVector.push_back(
                                memPortInfo(valName, getType(memVal.getType()), memVal, "bits", "dataIn"));
                    }
                } else if (memValPair.second == 2) {
                    if (resNum == 0 || resNum == 1) {
                        chisel_mem += "\t" + valName + " := false.B\n";
                    } else {
                        chisel_mem += "\t" + valName + " := DontCare\n";
                    }
                    if (resNum == 2) {
                        subOutVector.push_back(
                                memPortInfo(valName, getType(memVal.getType()), memVal, "bits", "writeIn"));
                        subOutVector.push_back(
                                memPortInfo(valName, getType(memVal.getType()), memVal, "bits", "readIn"));
                    } else if (resNum == 0) {
                        subOutVector.push_back(
                                memPortInfo(valName, getType(memVal.getType()), memVal, "valid", "writeIn"));
                    } else if (resNum == 3) {
                        subOutVector.push_back(
                                memPortInfo(valName, getType(memVal.getType()), memVal, "bits", "w_dataIn"));
                    } else if (resNum == 1) {
                        subOutVector.push_back(
                                memPortInfo(valName, getType(memVal.getType()), memVal, "valid", "readIn"));
                    }
                } else {
                    if (resNum == 0) {
                        chisel_mem += "\t" + valName + " := false.B\n";
                    } else {
                        chisel_mem += "\t" + valName + " := DontCare\n";
                    }
                    if (resNum == 1) {
                        subOutVector.push_back(
                                memPortInfo(valName, getType(memVal.getType()), memVal, "bits", "dataIn"));
                    } else if (resNum == 0) {
                        subOutVector.push_back(
                                memPortInfo(valName, getType(memVal.getType()), memVal, "valid", "dataIn"));
                    }
                }
            }
            auto &memInVector = memInPorts[comp];
            auto &subInVector = memInPortNames[get(comp.getName())];
            for (auto memValPair : memInVector) {
                auto memVal = memValPair.first;
                string valName = get_name(memVal);
                bool found = false;
                for (auto const &meet : subInVector) {
                    if (meet.portName == valName) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    continue;
                }
                chisel_mem += "\tval " + valName + " = IO(Input(" + getType(memVal.getType()) + "))\n";
                auto type = memVal.getType();
                subInVector.push_back(memPortInfo(valName, getType(memVal.getType()), memVal, "",
                                                  memValPair.second ? "readIn" : "dataOut"));
            }
//            std::cerr << chisel_mem << std::endl;
            return chisel_mem;
        }

        string dumpSTGComponent(hec::ComponentOp &comp) {
            string chisel_component = "class ";
            bool wrapped = comp.interfc() == "wrapped";
            string compName = get(comp.getName());
            chisel_component += compName + " extends MultiIOModule {\n";
            portNames[compName] = std::vector<string>();
            if (!wrapped) {
                chisel_component += "\tval go = IO(Input(Bool()))\n";
                chisel_component += "\tval done = IO(Output(Bool()))\n\tdone := 0.U\n";
            } else {

            }
            numInPorts[compName] = comp.numInPorts();
            for (auto val : comp.getArguments()) {
                if (!wrapped && val.getArgNumber() == comp.numInPorts() - 1) {
                    portNames[compName].push_back("go");
                    continue;
                } else if (!wrapped && val.getArgNumber() == comp.getNumArguments() - 1) {
                    portNames[compName].push_back("done");
                    continue;
                }
                if (!wrapped) {
                    portNames[compName].push_back(get_name(val));
                    chisel_component += "\tval " + get_name(val) + " = IO(";
                    chisel_component += val.getArgNumber() < comp.numInPorts() ? "Input(" : "Output(";
                    chisel_component += getType(val.getType());
                    chisel_component += "))\n";
                    if (val.getArgNumber() >= comp.numInPorts()) {
                        chisel_component += "\t" + get_name(val) + " := DontCare\n";
                    }
                } else {
                    string moduleName = get_name(val);
                    portNames[compName].push_back(moduleName + "_dyn");
                    chisel_component += "\tval " + moduleName + "_dyn = IO(";
                    chisel_component +=
                            val.getArgNumber() < comp.numInPorts() ? "Flipped(DecoupledIO(" : "DecoupledIO(";
                    chisel_component += getType(val.getType());
                    if (val.getArgNumber() < comp.numInPorts()) {
                        chisel_component += ")";
                    }
                    chisel_component += "))\n";
                    chisel_component += "\tval " + moduleName + " = " + moduleName + "_dyn.bits\n";
                    if (val.getArgNumber() >= comp.numInPorts()) {
                        chisel_component += "\t" + moduleName + " := DontCare\n";
                        chisel_component += "\t" + moduleName + "_dyn.valid := false.B\n";
                    }
                }
            }

            if (wrapped) {
                chisel_component += "\tval go = ";
                unsigned portNum = portNames[compName].size();
                bool first = true;
                for (unsigned idx = comp.numInPorts(); idx < portNum; ++idx) {
                    if (first) {
                        first = false;
                    } else {
                        chisel_component += " & ";
                    }
                    chisel_component += portNames[compName][idx] + ".ready";
                }
                for (unsigned idx = 0; idx < comp.numInPorts(); ++idx) {
                    if (first) {
                        first = false;
                    } else {
                        chisel_component += " & ";
                    }
                    chisel_component += portNames[compName][idx] + ".valid";
                }
                chisel_component += "\n";
                chisel_component += "\tdef done() : Unit = {\n";
                for (unsigned idx = comp.numInPorts(); idx < portNum; ++idx) {
                    chisel_component += "\t\t" + portNames[compName][idx] + ".valid := true.B\n";
                }
                chisel_component += "\t}\n";
            }
            //            comp->dump();
            for (auto &op : *(comp.getBody())) {
                if (auto hec = dyn_cast<hec::StateSetOp>(op)) {
                    chisel_component += generateMemPorts(comp);
                    chisel_component += dumpState(hec, wrapped);
                } else if (auto instance = dyn_cast<hec::InstanceOp>(op)) {
                    chisel_component += dumpInstance(instance, comp);
                } else if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
                    if (primitive.primitiveName() == "register") {
                        assert(primitive->getNumResults() == 1 && "Invalid register");
                        chisel_component += "\tval " + get_name(primitive.getResult(0));
                        chisel_component += " = Reg(" + getType(primitive.getType(0)) + ")\n";
                    } else {
                        insert_arithmetic_primitive(chisel_component, primitive, comp);
                    }
                } else if (auto constant = dyn_cast<ConstantOp>(op)) {
                } else {
                    op.dump();
                    assert(false && "Undefined operation in hec Component");
                }
            }
            if (wrapped) {
                unsigned portNum = portNames[compName].size();
                for (unsigned idx = 0; idx < comp.numInPorts(); ++idx) {
                    chisel_component += "\t" + portNames[compName][idx] + ".ready := ready\n";
                }
            }
            chisel_component += "}\n";
            return chisel_component;
        }
        
        void dummyComponent(hec::ComponentOp &comp) {
            string compName = get(comp.getName());
            portNames[compName] = std::vector<string>();
            numInPorts[compName] = comp.numInPorts();
            auto ports = hec::getComponentPortInfo(&*comp);
            for (auto val : comp.getArguments()) {
                set_name(val, get(ports[val.getArgNumber()].name.getValue()));
                portNames[compName].push_back(get(ports[val.getArgNumber()].name.getValue()));
            }
        }

        string dumpHandShakeComponent(hec::ComponentOp &comp) {
            string chisel_component = "class ";
//            string chisel_component = "";
            string compName = get(comp.getName());
//            auto hecDesign = cast<hec::DesignOp>(comp->getParentOp());
            chisel_component += compName + " extends MultiIOModule {\n";
            portNames[compName] = std::vector<string>();
//            chisel_component += "\tval go = IO(Input(Bool()))\n";
//            chisel_component += "\tval done = IO(Output(Bool()))\n\tdone := 0.U\n";
            numInPorts[compName] = comp.numInPorts();
            for (auto val : comp.getArguments()) {
                portNames[compName].push_back(get_name(val));
                //                val.dump();
                chisel_component += "\tval " + get_name(val) + " = IO(";
                chisel_component +=
                        val.getArgNumber() < comp.numInPorts() ? "Flipped(DecoupledIO(" : "DecoupledIO(";

                //                llvm::raw_string_ostream out(chisel_component);
                //                val.getType().print(out);
                chisel_component += getType(val.getType());
                chisel_component += val.getArgNumber() < comp.numInPorts() ? ")))\n" : "))\n";
                if (val.getArgNumber() >= comp.numInPorts()) {
                    chisel_component += "\t" + get_name(val) + " := DontCare\n";
                }
            }
            //            comp->dump();
            for (auto &op : *(comp.getBody())) {
                if (auto graph = dyn_cast<hec::GraphOp>(op)) {
                    chisel_component += dumpGraph(graph);
                } else if (auto instance = dyn_cast<hec::InstanceOp>(op)) {
                    chisel_component += dumpInstance(instance, comp);
                } else if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
                    //                    std::cerr << get(primitive.instanceName()) << std::endl;
                    //                    std::cerr << get(primitive.primitiveName()) << std::endl;
                    if (primitive.primitiveName() == "register") {
                        assert(primitive->getNumResults() == 1 && "Invalid register");
                        chisel_component += "\tval " + get_name(primitive.getResult(0));
                        chisel_component += " = Reg(" + getType(primitive.getType(0)) + ")\n";
                    } else {
                        string moduleName = get(primitive.instanceName());
                        chisel_component += "\tval " + moduleName + " = Module(new ";
                        //Fixme : Consider about integer width in primitive operations
                        string primName = get(primitive.primitiveName());
                        if (primName == "add_integer") {
                            chisel_component +=
                                    "AddIDynamic(" + std::to_string(getWidth(primitive.getResult(0).getType())) + ")";
                        } else if (primName == "sub_integer") {
                            chisel_component +=
                                    "SubIDynamic(" + std::to_string(getWidth(primitive.getResult(0).getType())) + ")";
                        } else if (primName == "and") {
                            chisel_component +=
                                    "AndDynamic(" + std::to_string(getWidth(primitive.getResult(0).getType())) + ")";
                        } else if (primName == "shift_left") {
                            chisel_component +=
                                    "ShiftLeftDynamic(" + std::to_string(getWidth(primitive.getResult(0).getType())) +
                                    ")";
                        } else if (primName == "trunc_integer") {
                            chisel_component +=
                                    "TruncIDynamic(" + std::to_string(getWidth(primitive.getResult(0).getType())) +
                                    "," + std::to_string(getWidth(primitive.getResult(1).getType())) + ")";
                        } else if (primName == "mul_integer") {
                            chisel_component +=
                                    "MulIDynamic(" + std::to_string(getWidth(primitive.getResult(0).getType())) + ")";
                        } else if (primName == "div_integer") {
                            chisel_component +=
                                    "DivIDynamic(" + std::to_string(getWidth(primitive.getResult(0).getType())) + ")";
                        } else if (primName == "merge") {
                            chisel_component +=
                                    "Merge(" + std::to_string(getWidth(primitive.getResult(0).getType())) + ")()";
                        } else if (primName == "select") {
                            chisel_component +=
                                    "Select(" + std::to_string(getWidth(primitive.getResult(1).getType())) + ")";
                        } else if (primName == "buffer") {
                            chisel_component +=
                                    "ElasticBuffer(" + std::to_string(getWidth(primitive.getResult(0).getType())) + ")";
                        } else if (primName.find("fork") != string::npos) {
                            if (primName.find(":") != string::npos) {
                                chisel_component +=
                                        "Fork(" + std::to_string(getWidth(primitive.getResult(0).getType())) + ")(" +
                                        primName.substr(primName.find(":") + 1) + ")";
                            } else {
                                chisel_component +=
                                        "Fork(" + std::to_string(getWidth(primitive.getResult(0).getType())) + ")()";
                            }
                        } else if (primName == "branch") {
                            chisel_component +=
                                    "Branch(" + std::to_string(getWidth(primitive.getResult(1).getType())) + ")";
                        } else if (primName == "mux_dynamic") {
                            chisel_component +=
                                    "MuxDynamic(" + std::to_string(getWidth(primitive.getResult(0).getType())) + ")()";
                        } else if (primName == "control_merge") {
                            chisel_component +=
                                    "Control_Merge(" + std::to_string(getWidth(primitive.getResult(0).getType())) + ")";
                        } else if (primName.find("cmp_integer") != string::npos) {
                            if (primName.find("sge") != string::npos) {
                                chisel_component += "GreaterthanIDynamic(" +
                                                    std::to_string(getWidth(primitive.getResult(0).getType())) + ")";
                            } else if (primName.find("sle") != string::npos) {
                                chisel_component += "LessEqualthanIDynamic(" +
                                                    std::to_string(getWidth(primitive.getResult(0).getType())) + ")";
                            } else if (primName.find("ne") != string::npos) {
                                chisel_component += "NotEqualIDynamic(" +
                                                    std::to_string(getWidth(primitive.getResult(0).getType())) + ")";
                            } else {
                                std::cerr << primName << std::endl;
                                assert(false && "Undefined compare predicate");
                            }
                        } else if (primName.find("dyn_Mem") != string::npos) {
                            string memName = primName;
                            string loadnum = memName.substr(memName.find(":") + 1,
                                                            memName.find(",") - memName.find(":") - 1);
                            string storenum = memName.substr(memName.find(",") + 1,
                                                             memName.find("#") - memName.find(",") - 1);
                            string memSize = memName.substr(memName.find("#") + 1);
//                            std::cerr << memName << "!!" << loadnum << "??" << storenum << std::endl;
                            chisel_component += "DynMem(" + loadnum + "," + storenum + ")(" + memSize + "," +
                                                std::to_string(getWidth(primitive.getResult(1).getType())) + ")";
                        } else if (primName.find("load") != string::npos) {
                            string memName = primName;
                            string memSize = memName.substr(memName.find("#") + 1);
                            chisel_component += "Load(" + memSize + "," +
                                                std::to_string(getWidth(primitive.getResult(1).getType())) + ")";
                        } else if (primName.find("store") != string::npos) {
                            string memName = primName;
                            string memSize = memName.substr(memName.find("#") + 1);
                            chisel_component += "Store(" + memSize + "," +
                                                std::to_string(getWidth(primitive.getResult(1).getType())) + ")";
                        } else if (primName == "fptosi") {
                            auto type = primitive.getResult(0).getType();
                            if (type.isa<Float64Type>()) {
                                chisel_component += "FloatToIntDynamic(9, " +
                                                    std::to_string(getWidth(primitive.getResult(1).getType())) +
                                                    ", 11, 53)";
                            } else if (type.isa<Float32Type>()) {
                                chisel_component += "FloatToIntDynamic(9, " +
                                                    std::to_string(getWidth(primitive.getResult(1).getType())) +
                                                    ", 8, 24)";
                            }
                        } else if (primName == "mul_float") {
                            auto type = primitive.getResult(2).getType();
                            if (type.isa<Float64Type>()) {
                                chisel_component += "MulFDynamic(9, 11, 53)";
                            } else if (type.isa<Float32Type>()) {
                                chisel_component += "MulFDynamic(4, 8, 24)";
                            }
                        } else if (primName == "neg_float") {
                            auto type = primitive.getResult(1).getType();
                            if (type.isa<Float64Type>()) {
                                chisel_component += "NegFDynamic(11, 53)";
                            } else if (type.isa<Float32Type>()) {
                                chisel_component += "NegFDynamic(8, 24)";
                            }
                        } else if (primName == "div_float") {
                            auto type = primitive.getResult(2).getType();
                            if (type.isa<Float64Type>()) {
                                chisel_component += "DivFDynamic(9, 11, 53)";
                            } else if (type.isa<Float32Type>()) {
                                chisel_component += "DivFDynamic(4, 8, 24)";
                            }
                        } else if (primName.find("cmp_float") != string::npos) {
                            auto type = primitive.getResult(0).getType();
                            if (type.isa<Float64Type>()) {
                                chisel_component += "CmpFDynamic(2, 11, 53)";
                            } else if (type.isa<Float32Type>()) {
                                chisel_component += "CmpFDynamic(2, 8, 24)";
                            }
                            string cmp_type = primName.substr(primName.rfind("_") + 1);
                            if (cmp_type == "ult") {
                                chisel_component += "(4.U)";
                            } else if (cmp_type == "ule") {
                                chisel_component += "(5.U)";
                            } else if (cmp_type == "ugt") {
                                chisel_component += "(2.U)";
                            } else if (cmp_type == "uge") {
                                chisel_component += "(3.U)";
                            } else {
                                std::cerr << cmp_type << std::endl;
                                assert(false && "not found CmpF");
                            }
                        } else if (primName == "add_float") {
                            auto type = primitive.getResult(2).getType();
                            if (type.isa<Float64Type>()) {
                                chisel_component += "AddFDynamic(13, 11, 53)";
                            } else if (type.isa<Float32Type>()) {
                                chisel_component += "AddFDynamic(8, 8, 24)";
                            }
                        } else if (primName == "sub_float") {
                            auto type = primitive.getResult(2).getType();
                            if (type.isa<Float64Type>()) {
                                chisel_component += "SubFDynamic(13, 11, 53)";
                            } else if (type.isa<Float32Type>()) {
                                chisel_component += "SubFDynamic(8, 8, 24)";
                            }
                        } else if (primName == "constant") {
                            chisel_component +=
                                    "Constant(" + std::to_string(getWidth(primitive.getResult(1).getType())) + ")";
                        } else if (primName.find("fifo") != string::npos) {
                            chisel_component += "ElasticFIFO(" + primName.substr(primName.find(":") + 1) + "," +
                                                std::to_string(getWidth(primitive.getResult(1).getType())) + ")";
                        } else {
                            std::cerr << get(primitive.primitiveName()) << std::endl;
                            assert(false && "Unknown primitive operation");
                        }
                        chisel_component += ")\n";
                        auto portInfos = primitive.getPrimitivePortInfo();
                        for (auto val : primitive.getResults()) {
                            chisel_component += "\tval " + get_name(val);
                            chisel_component += " = " + moduleName + ".";
                            string portName = get(portInfos[val.getResultNumber()].name.getValue());
                            if (portName.find(".") != string::npos) {
                                portName = portName.replace(portName.find("."), 1, " apply ");
                            }
                            chisel_component += portName + "\n";
                            chisel_component += "\t" + get_name(val) + " := DontCare\n";
                        }
                        //                        std::cerr << get(primitive.primitiveName()) << std::endl;
                        //                        assert(false && "Unknown primitive operation");
                    }
                } else if (auto constant = dyn_cast<ConstantOp>(op)) {
                } else {
                    op.dump();
                    assert(false && "Undefined operation in hec Component");
                }
            }
            for (auto &op : *(comp.getBody())) {
                if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
                    auto portInfo = primitive.getPrimitivePortInfo();
                    for (unsigned idx = 0; idx < portInfo.size(); ++idx) {
                        if (portInfo[idx].direction == hec::PortDirection::OUTPUT) {
                            auto port = primitive.getResult(idx);
                            std::vector<hec::AssignOp> assign_set;
                            int use_count = 0;
                            for (auto &bval : port.getUses()) {
                                if (auto assign = dyn_cast<hec::AssignOp>(bval.getOwner())) {
                                    if (assign.src() == port) {
                                        assign_set.push_back(assign);
                                        ++use_count;
                                    }
                                }
                            }
                            if (use_count == 0) {
//                                std::cerr << primitive.primitiveName().str() << "SINK";
//                                primitive.dump();
//                                std::cerr << portInfo[idx].name.getValue().str() << ":";
//                                port.dump();
                                chisel_component += "\t" + get_name(port) + ".ready := true.B\n";
                            }
                        }
                    }
                }
            }
            for (auto &op : *(comp.getBody())) {
                if (auto instance = dyn_cast<hec::InstanceOp>(op)) {
                    string instanceName = instance.instanceName().str();
                    string componentName = instance.componentName().str();
                    auto &allMemOutPorts = DUMP::memOutPortNames[componentName];
                    for (auto memPort : allMemOutPorts) {
                        //FIXME: Deal with arbiter number
                        auto memory = cast<hec::PrimitiveOp>(memPort.val.getDefiningOp());
                        string name = DUMP::get(memory.instanceName()) + memPort.name + memPort.data;
                        int arbiterNum = DUMP::arbiterNums[name];
                        DUMP::arbiterNums[name] = arbiterNum + 1;
                        chisel_component += "\t" + DUMP::get(memory.instanceName()) + "." + memPort.name + "(" +
                                            std::to_string(arbiterNum) + ").";
                        chisel_component += memPort.data + " := " + instanceName + "." + memPort.portName + "\n";
                    }
                    auto &allMemInPorts = DUMP::memInPortNames[componentName];
                    for (auto memPort : allMemInPorts) {
                        chisel_component +=
                                "\t" + instanceName + "." + memPort.portName + " := " + memPort.portName + "\n";
                    }
//                    chisel_component += dumpInstance(instance, comp);
                }
            }
            chisel_component += "}\n";
            return chisel_component;
        }

        string dumpPipelineFunctionComponent(hec::ComponentOp &comp) {
            bool wrapped = comp.interfc() == "wrapped";
            string chisel_component = "class ";
            string compName = get(comp.getName());
            chisel_component += compName + " extends MultiIOModule {\n";
            portNames[compName] = std::vector<string>();
            chisel_component += "\tval new_input = Wire(Bool())\n"
                                "new_input := false.B\n";
            chisel_component += "\tval done = Wire(Bool())\n"
                                "\tdone := false.B\n";
//            chisel_component += "\tval start = IO(Input(Bool()))\n";
            //Useless
            chisel_component += "\tval new_output = IO(Output(Bool()))\n\tnew_output := 0.U\n";
            numInPorts[compName] = comp.numInPorts();
            if (!wrapped) {
                for (auto val : comp.getArguments()) {
                    if (val.getArgNumber() == comp.numInPorts() - 1) {
                        portNames[compName].push_back("go");
                        continue;
                    } else if (val.getArgNumber() == comp.getNumArguments() - 1) {
                        portNames[compName].push_back("done");
                        continue;
                    }
                    portNames[compName].push_back(get_name(val));
                    chisel_component += "\tval " + get_name(val) + " = IO(";
                    chisel_component += val.getArgNumber() < comp.numInPorts() ? "Input(" : "Output(";
                    chisel_component += getType(val.getType());
                    chisel_component += "))\n";
                    if (val.getArgNumber() >= comp.numInPorts()) {
                        chisel_component += "\t" + get_name(val) + " := DontCare\n";
                    }
                }
            } else {
                string all_valid = "\tval all_valid = ";
                string ce_signal = "\tval continue = ";
                bool first_in = true;
                bool first_out = true;
                for (auto val : comp.getArguments()) {
                    //FIXME: go & done signals
                    if (val.getArgNumber() == comp.numInPorts() - 1) {
                        portNames[compName].push_back("go");
                        continue;
                    } else if (val.getArgNumber() == comp.getNumArguments() - 1) {
                        portNames[compName].push_back("done");
                        continue;
                    }
                    string val_name = get_name(val);
                    portNames[compName].push_back(val_name);
                    chisel_component += "\tval d_" + val_name + " = IO(";
                    chisel_component += val.getArgNumber() < comp.numInPorts() ?
                                        "Flipped(DecoupledIO(" :
                                        "DecoupledIO(";
                    chisel_component += getType(val.getType());
                    chisel_component += val.getArgNumber() < comp.numInPorts() ?
                                        ")))\n" : "))\n";
                    chisel_component += "\tval " + val_name + " = d_" + val_name + ".bits\n";
                    if (val.getArgNumber() >= comp.numInPorts()) {
                        chisel_component += "\t" + val_name + " := DontCare\n";
                        chisel_component += "\td_" + val_name + ".valid := false.B\n";
                        if (!first_out) {
                            ce_signal += " & ";
                        }
                        first_out = false;
                        ce_signal += "d_" + val_name + ".ready";
                    } else {
                        chisel_component += "\td_" + val_name + ".ready := false.B\n";
                        if (!first_in) {
                            all_valid += " & ";
                        }
                        first_in = false;
                        all_valid += "d_" + val_name + ".valid";
                    }
                }
                chisel_component += all_valid + "\n";
                chisel_component += ce_signal + "\n";
                chisel_component += "\tval next_continue = RegNext(continue)\n";
                chisel_component += "\tval go = all_valid\n";
            }

            int II = comp->getAttr("II").cast<IntegerAttr>().getInt();
            int latency = comp->getAttr("latency").cast<IntegerAttr>().getInt();
            for (auto &op : *(comp.getBody())) {
                if (auto hec = dyn_cast<hec::StageSetOp>(op)) {
                    chisel_component += generateMemPorts(comp);
                    chisel_component += dumpStage(hec, II, latency, true);
                } else if (auto instance = dyn_cast<hec::InstanceOp>(op)) {
                    chisel_component += dumpInstance(instance, comp);
                } else if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
                    //                    std::cerr << get(primitive.instanceName()) << std::endl;
                    //                    std::cerr << get(primitive.primitiveName()) << std::endl;
                    if (primitive.primitiveName() == "register") {
                        assert(primitive->getNumResults() == 1 && "Invalid register");
                        chisel_component += "\tval " + get_name(primitive.getResult(0));
                        chisel_component += " = Reg(" + getType(primitive.getType(0)) + ")\n";
                    } else {
                        insert_arithmetic_primitive(chisel_component, primitive, comp, true);
                    }
                } else if (auto constant = dyn_cast<ConstantOp>(op)) {
                } else if (auto init = dyn_cast<hec::InitOp>(op)) {
                    chisel_component += "\twhen (go & continue) {\n";
                    chisel_component +=
                            "\t\t" + get_name(init.dst()) + " := " + get_name(init.src()) + "\n\t}\n";
                } else {
                    op.dump();
                    assert(false && "Undefined operation in hec Component");
                }
            }
            chisel_component += "\twhen (continue) {\n"
                                "\t\twhen (counter === 0.U) {\n";
            for (auto val : comp.getArguments()) {
                //FIXME: go & done signals
                if (val.getArgNumber() == comp.numInPorts() - 1) {
                    continue;
                } else if (val.getArgNumber() == comp.getNumArguments() - 1) {
                    continue;
                }
                string val_name = get_name(val);
                if (val.getArgNumber() >= comp.numInPorts()) {
                } else {
                    chisel_component += "\t\t\td_" + get_name(val) + ".ready := all_valid\n";
                }
            }
            chisel_component += "\t\t}\n";
            for (auto val : comp.getArguments()) {
                //FIXME: go & done signals
                if (val.getArgNumber() == comp.numInPorts() - 1) {
                    continue;
                } else if (val.getArgNumber() == comp.getNumArguments() - 1) {
                    continue;
                }
                string val_name = get_name(val);
                if (val.getArgNumber() >= comp.numInPorts()) {
                    chisel_component += "\t\td_" + get_name(val) + ".valid := done\n";
                } else {
                }
            }
            chisel_component += "\t}\n";
            chisel_component += "}\n";
            return chisel_component;
        }

        string dumpPipelineForComponent(hec::ComponentOp &comp) {
            bool wrapped = comp.interfc() == "wrapped";
            string chisel_component = "class ";
            string compName = get(comp.getName());
            chisel_component += compName + " extends MultiIOModule {\n";
            portNames[compName] = std::vector<string>();
            if (!wrapped) {
                chisel_component += "\tval go = IO(Input(Bool()))\n";
            }
            chisel_component += "\tval start = RegInit(false.B)\n";
            chisel_component += "\tval new_input = Wire(Bool())\n"
                                "\tnew_input := false.B\n";
            if (!wrapped) {
                chisel_component += "\tval done = IO(Output(Bool()))\n\tdone := 0.U\n";
            } else {
                chisel_component += "\tval done = Wire(Bool())\n\tdone := 0.U\n";
            }
            numInPorts[compName] = comp.numInPorts();
            if (!wrapped) {
                for (auto val : comp.getArguments()) {
                    if (val.getArgNumber() == comp.numInPorts() - 1) {
                        portNames[compName].push_back("go");
                        continue;
                    } else if (val.getArgNumber() == comp.getNumArguments() - 1) {
                        portNames[compName].push_back("done");
                        continue;
                    }
                    portNames[compName].push_back(get_name(val));
                    chisel_component += "\tval " + get_name(val) + " = IO(";
                    chisel_component += val.getArgNumber() < comp.numInPorts() ? "Input(" : "Output(";
                    chisel_component += getType(val.getType());
                    chisel_component += "))\n";
                    if (val.getArgNumber() >= comp.numInPorts()) {
                        chisel_component += "\t" + get_name(val) + " := DontCare\n";
                    }
                }
            } else {
                string all_valid = "\tval all_valid = ";
                string go = "\tval go = all_valid";
                bool first = true;
                for (auto val : comp.getArguments()) {
                    //FIXME: go & done signals
                    if (val.getArgNumber() == comp.numInPorts() - 1) {
                        portNames[compName].push_back("go");
                        continue;
                    } else if (val.getArgNumber() == comp.getNumArguments() - 1) {
                        portNames[compName].push_back("done");
                        continue;
                    }
                    string val_name = get_name(val);
                    portNames[compName].push_back(val_name);
                    chisel_component += "\tval d_" + val_name + " = IO(";
                    chisel_component += val.getArgNumber() < comp.numInPorts() ?
                                        "Flipped(DecoupledIO(" :
                                        "DecoupledIO(";
                    chisel_component += getType(val.getType());
                    chisel_component += val.getArgNumber() < comp.numInPorts() ?
                                        ")))\n" : "))\n";
                    chisel_component += "\tval " + val_name + " = d_" + val_name + ".bits\n";
                    if (val.getArgNumber() >= comp.numInPorts()) {
                        chisel_component += "\t" + get_name(val) + " := DontCare\n";
                        go += " & d_" + val_name + ".ready";
                    } else {
                        if (!first) {
                            all_valid += " & ";
                        }
                        all_valid += "d_" + val_name + ".valid";
                        first = false;
                    }
                }
                chisel_component += all_valid + "\n";
                chisel_component += go + "\n";

            }

            int II = comp->getAttr("II").cast<IntegerAttr>().getInt();
            int latency;
            for (auto &op : *(comp.getBody())) {
                if (auto hec = dyn_cast<hec::StageSetOp>(op)) {
                    latency = hec.getBody()->getOperations().size() - 1;
                }
            }
            for (auto &op : *(comp.getBody())) {
                if (auto hec = dyn_cast<hec::StageSetOp>(op)) {
                    chisel_component += generateMemPorts(comp);
                    chisel_component += dumpStage(hec, II, latency);
                } else if (auto instance = dyn_cast<hec::InstanceOp>(op)) {
                    chisel_component += dumpInstance(instance, comp);
                } else if (auto wire = dyn_cast<hec::WireOp>(op)) {
                    if (wire.name() == "i") {
                        string index = get_name(wire.out());
                        chisel_component += "\tval " + index;
                        chisel_component += " = Reg(" + getType(comp.getArgument(0).getType()) + ")\n";
                        chisel_component +=
                                "\twhen (go) {\n\t\t" + index + " := " + get_name(comp.getArgument(0));
                        chisel_component += "\n\t}\n";
                    } else {
                        assert(false && "Unknown wire operation");
                    }
                } else if (auto init = dyn_cast<hec::InitOp>(op)) {
                    chisel_component += "\twhen (go) {\n";
                    chisel_component +=
                            "\t\t" + get_name(init.dst()) + " := " + get_name(init.src()) + "\n\t}\n";
                } else if (auto primitive = dyn_cast<hec::PrimitiveOp>(op)) {
                    //                    std::cerr << get(primitive.instanceName()) << std::endl;
                    //                    std::cerr << get(primitive.primitiveName()) << std::endl;
                    if (primitive.primitiveName() == "register") {
                        assert(primitive->getNumResults() == 1 && "Invalid register");
                        chisel_component += "\tval " + get_name(primitive.getResult(0));
                        chisel_component += " = Reg(" + getType(primitive.getType(0)) + ")\n";
                    } else {
                        insert_arithmetic_primitive(chisel_component, primitive, comp);
                    }
                } else if (auto constant = dyn_cast<ConstantOp>(op)) {
                } else {
                    op.dump();
                    assert(false && "Undefined operation in hec Component");
                }
            }
            for (auto &op : *(comp.getBody())) {
                if (auto wire = dyn_cast<hec::WireOp>(op)) {
                    if (wire.name() == "i") {
                        string index = get_name(wire.out());
                        chisel_component +=
                                "\tval ub_reg = Reg(" + getType(comp.getArgument(1).getType()) + ")\n";
                        chisel_component += "\twhen (go) {\n"
                                            "\t\tub_reg := " + get_name(comp.getArgument(1)) +
                                            "\n\t}\n";
                        chisel_component +=
                                "\tval step_reg = Reg(" + getType(comp.getArgument(2).getType()) + ")\n";
                        chisel_component += "\twhen (go) {\n"
                                            "\t\tstep_reg := " + get_name(comp.getArgument(2)) +
                                            "\n\t}\n";
                        chisel_component +=
                                "\tval upper_bound = Mux(go, " + get_name(comp.getArgument(1)) +
                                ", ub_reg)\n";
                        chisel_component +=
                                "\tval step = Mux(go, " + get_name(comp.getArgument(2)) + ", step_reg)\n";
                        if (II == 1) {
                            chisel_component += "\tnew_input := start\n";
                            chisel_component += "\twhen (start) {\n";
                            chisel_component += "\t\twhen (" + index + " <= upper_bound) {\n";
                            chisel_component += "\t\t\t" + index + " := " + index + " + step\n";
                            chisel_component += "\t\t}.otherwise {\n"
                                                "\t\t\tstart := false.B\n"
                                                "\t\t\tnew_input := false.B\n"
                                                "\t\t}\n";
                            chisel_component += "\t}\n";
                        } else {
                            chisel_component += "\tnew_input := start\n";
                            chisel_component += "\twhen (start) {\n";
                            chisel_component += "\t\twhen (" + index + " <= upper_bound) {\n";
                            chisel_component +=
                                    "\t\t\twhen (counter === " + std::to_string(II - 1) + ".U) {\n";
                            chisel_component += "\t\t\t\t" + index + " := " + index + " + step\n\t\t\t}\n";
                            chisel_component += "\t\t}.otherwise {\n"
                                                "\t\t\tstart := false.B\n"
                                                "\t\t\tnew_input := false.B\n"
                                                "\t\t}\n";
                            chisel_component += "\t}\n";
                        }

                        if (wrapped) {
                            chisel_component += "\tval init = RegInit(true.B)\n"
                                                "\twhen (done) {\n"
                                                "\t\tinit := true.B\n"
                                                "\t}\n"
                                                "\twhen (new_input) {\n"
                                                "\t\tinit := false.B\n"
                                                "\t}\n";

                            for (auto val : comp.getArguments()) {
                                //FIXME: go & done signals
                                if (val.getArgNumber() == comp.numInPorts() - 1) {
                                    continue;
                                } else if (val.getArgNumber() == comp.getNumArguments() - 1) {
                                    continue;
                                }
                                string val_name = get_name(val);
                                if (val.getArgNumber() >= comp.numInPorts()) {
                                    chisel_component += "\td_" + val_name + ".valid := done\n";
                                } else {
                                    chisel_component += "\td_" + val_name + ".ready := all_valid & init\n";
                                }
                            }
                        }
                    } else {
                        assert(false && "Unknown wire operation");
                    }
                }
            }
            chisel_component += "}\n";
            return chisel_component;
        }
    }

    struct DumpChiselPass : public dumpChiselBase<DumpChiselPass> {
        void runOnOperation() override {
            mlir::ModuleOp m = getOperation();
            std::string chisel_code = "";
            bool found_dynamic = false;
            hec::DesignOp hecDesign;
            for (auto &module : *(m.getBody())) {
                if (isa<hec::DesignOp>(module)) {
                    hecDesign = cast<hec::DesignOp>(module);
                    for (auto &func : *(hecDesign.getBody())) {
                        if (auto comp = dyn_cast<hec::ComponentOp>(func)) {
                            if (comp.style() == "STG") {
                                chisel_code += DUMP::dumpSTGComponent(comp);
                            } else if (comp.style() == "handshake") {
                                //assert(!found_dynamic);
                                found_dynamic = true;
                                chisel_code += DUMP::dumpHandShakeComponent(comp);
                            } else if (comp.style() == "pipeline") {
                                if (comp->getAttr("pipeline").cast<StringAttr>().getValue() == "func") {
                                    chisel_code += DUMP::dumpPipelineFunctionComponent(comp);
                                } else {
                                    chisel_code += DUMP::dumpPipelineForComponent(comp);
                                }
                            } else if (comp.style() == "dummy") {
                                DUMP::dummyComponent(comp);
                            }
                        } else if (auto primitive = dyn_cast<hec::PrimitiveOp>(func)) {
                            std::string primName = DUMP::get(primitive.primitiveName());
                            if (primName == "mem") {
                                std::string memName = DUMP::get(primitive.instanceName());
                                auto len = primitive->getAttr("len").cast<IntegerAttr>().getInt();
                                auto type = DUMP::get(
                                        primitive->getAttr("ports").cast<StringAttr>().getValue());
                                int intType;
                                int use_num = 0;
                                std::set<std::string> use_set;
                                //FIXME: Instance num
                                for (auto &val : primitive.getResult(0).getUses()) {
                                    use_set.insert(DUMP::get_component_op(val.getOwner()).getName().str());
                                }
                                use_num = use_set.size();
                                if (type == "r") {
                                    chisel_code +=
                                            "\tval " + memName + " = Module(new ReadMem(" + std::to_string(len) + ", " +
                                            std::to_string(DUMP::getWidth(primitive.getResult(2).getType())) + ", " +
                                            std::to_string(use_num) + "))\n";
                                    intType = 0;
                                } else if (type == "w") {
                                    chisel_code +=
                                            "\tval " + memName + " = Module(new WriteMem(" + std::to_string(len) +
                                            ", " + std::to_string(DUMP::getWidth(primitive.getResult(2).getType())) +
                                            ", " + std::to_string(use_num) + "))\n";
                                    intType = 1;
                                } else {
                                    chisel_code +=
                                            "\tval " + memName + " = Module(new ReadWriteMem(" + std::to_string(len) +
                                            ", " + std::to_string(DUMP::getWidth(primitive.getResult(3).getType())) +
                                            ", " + std::to_string(use_num) + "))\n";
                                    intType = 2;
                                }
                                auto portInfos = primitive.getPrimitivePortInfo();
                                for (auto val : primitive.getResults()) {
                                    chisel_code += "\tval " + DUMP::get_name(val);
                                    chisel_code += " = " + memName + ".";
                                    chisel_code += DUMP::get(portInfos[val.getResultNumber()].name.getValue()) + "\n";
                                    if (portInfos[val.getResultNumber()].direction == hec::PortDirection::OUTPUT) {
                                        for (auto &bval : val.getUses()) {
                                            auto subComp = DUMP::get_component_op(bval.getOwner());
                                            auto &memVector = DUMP::memInPorts[subComp];
                                            bool found = false;
                                            for (auto const &memPort : memVector) {
                                                if (memPort.first == val) {
                                                    found = true;
                                                    break;
                                                }
                                            }
                                            if (!found) {
                                                DUMP::memInPorts[subComp].push_back(std::make_pair(val, intType));
                                            }
                                        }
                                    } else {
                                        for (auto &bval : val.getUses()) {
                                            auto subComp = DUMP::get_component_op(bval.getOwner());
                                            auto &memVector = DUMP::memOutPorts[subComp];
                                            bool found = false;
                                            for (auto const &memPort : memVector) {
                                                if (memPort.first == val) {
                                                    found = true;
                                                    break;
                                                }
                                            }
                                            if (!found) {
                                                DUMP::memOutPorts[subComp].push_back(std::make_pair(val, intType));
                                            }
                                        }
                                    }
                                }
//                                chisel_code += "val " + memName + " = Module(new "
                            } else {
                                primitive->dump();
                                assert(false && "Invalid primitive operation");
                            }
                        }
                    }
                }
            }

            if (!found_dynamic) {
                for (auto &module : *(m.getBody())) {
                    if (auto hecDesign = dyn_cast<hec::DesignOp>(module)) {
                        chisel_code =
                                "class " + DUMP::get(hecDesign.symbol()) + " extends MultiIOModule {\n" + chisel_code;
                    }
                }
                chisel_code += "\tval main = Module(new main)\n";
                auto &allMemOutPorts = DUMP::memOutPortNames["main"];
                for (auto memPort : allMemOutPorts) {
                    //FIXME: Deal with arbiter number
                    auto memory = cast<hec::PrimitiveOp>(memPort.val.getDefiningOp());
                    std::string name = DUMP::get(memory.instanceName()) + memPort.name + memPort.data;
                    int arbiterNum = DUMP::arbiterNums[name];
                    DUMP::arbiterNums[name] = arbiterNum + 1;
                    chisel_code += "\t" + DUMP::get(memory.instanceName()) + "." + memPort.name + "(" +
                                   std::to_string(arbiterNum) + ").";
                    chisel_code += memPort.data + " := main." + memPort.portName + "\n";
                }
                chisel_code += "\tval go = IO(Input(Bool()))\n"
                               "\tmain.go := go\n"
                               "\tval done = IO(Output(Bool()))\n"
                               "\tdone := main.done\n";
                auto &allMemInPorts = DUMP::memInPortNames["main"];
                for (auto memPort : allMemInPorts) {
                    std::string global_name = memPort.portName;
                    if (global_name.find("_") != std::string::npos) {
                        global_name = global_name.substr(global_name.rfind("_") + 1);
                    }
                    chisel_code += "\tmain." + memPort.portName + " := " + global_name + "\n";
                }
                for (auto &func : *(hecDesign.getBody())) {
                    if (auto primitive = dyn_cast<hec::PrimitiveOp>(func)) {
                        std::string primName = DUMP::get(primitive.primitiveName());
                        if (primName == "mem") {
                            std::string memName = DUMP::get(primitive.instanceName());
                            auto len = primitive->getAttr("len").cast<IntegerAttr>().getInt();
                            auto type = DUMP::get(
                                    primitive->getAttr("ports").cast<StringAttr>().getValue());
                            if (type != "w") continue;
                            chisel_code += "\tval " + memName + "_test_addr = IO(Input(UInt(log2Ceil(";
                            chisel_code += std::to_string(len) + ").W)))\n";
                            chisel_code += "\t" + memName + ".test_addr := " + memName + "_test_addr\n";
                            chisel_code += "\tval " + memName + "_test_data = IO(Output(UInt(" +
                                           std::to_string(DUMP::getWidth(primitive.getResult(2).getType()));
                            chisel_code += ".W)))\n"
                                           "\t" + memName + "_test_data := " + memName + ".test_data\n";
                            chisel_code += "\t" + memName + ".finished := done\n";
                        }
                    }
                }
            } else {
                chisel_code = "class " + DUMP::get(hecDesign.symbol()) + " extends MultiIOModule {\n" + chisel_code;
		chisel_code += "\tval main = Module(new main)\n";
                m.walk([&](hec::ComponentOp op) {
                    if (op.getName() != "main") return;
                    auto &ports = DUMP::portNames["main"];
                    for (unsigned idx = 0; idx != op.getNumArguments(); ++idx) {
                        if (idx < op.numInPorts()) {
                            chisel_code += "\tval " + ports[idx] + " = IO(Flipped(DecoupledIO(" +
                                           DUMP::getType(op.getArgument(idx).getType()) + ")))\n";
                            chisel_code += "\tmain." + ports[idx] + " <> " + ports[idx] + "\n";
                        } else {
                            chisel_code += "\tval " + ports[idx] + " = IO(DecoupledIO(" +
                                           DUMP::getType(op.getArgument(idx).getType()) + "))\n";
                            chisel_code += "\t" + ports[idx] + " <> main." + ports[idx] + "\n";
                        }
                    }
                });

                chisel_code += "\tval finish = IO(Input(Bool()))\n";
                for (auto &component : *(hecDesign.getBody())) {
                    if (auto hecComponent = dyn_cast<hec::ComponentOp>(component)) {
                        for (auto &func : *(hecComponent.getBody())) {
                            if (auto primitive = dyn_cast<hec::PrimitiveOp>(func)) {
                                std::string primName = DUMP::get(primitive.primitiveName());
                                if (primName.find("dyn_Mem") != std::string::npos) {
                                    std::string memName = DUMP::get(primitive.instanceName());
                                    chisel_code += "\t" + memName + ".read_address := DontCare\n";
                                    chisel_code += "\t" + memName + ".finish := DontCare\n";
                                    std::string loadnum = primName.substr(primName.find(":") + 1,
                                                                          primName.find(",") -
                                                                          primName.find(":") - 1);
                                    std::string storenum = primName.substr(primName.find(",") + 1,
                                                                           primName.find("#") -
                                                                           primName.find(",") - 1);
                                    std::string len = primName.substr(primName.find("#") + 1);
                                    if (storenum != "0") {
                                        chisel_code += "\t" + memName + ".finish := finish\n";
                                        chisel_code += "\tval " + memName + "_addr = IO(Input(UInt(log2Ceil(";
                                        chisel_code += len + ").W)))\n";
                                        chisel_code +=
                                                "\t" + memName + ".read_address := " + memName + "_addr\n";
                                        chisel_code += "\tval " + memName + "_data = IO(Output(UInt(";
                                        chisel_code += std::to_string(
                                                DUMP::getWidth(primitive.getResult(1).getType()));
                                        chisel_code += ".W)))\n";
                                        chisel_code += "\t" + memName + "_data := " + memName + ".out_data\n";
                                    }

//                                    std::string len = primName.substr(primName.find("#"));
//                                    chisel_code +=
//                                            "\tval " + memName + "_test_addr = IO(Input(UInt(log2Ceil(";
//                                    chisel_code += len + ").W)))\n";
//                                    chisel_code += "\tval " + memName + "_test_data = IO(Output(UInt(" +
//                                                   std::to_string(
//                                                           DUMP::getWidth(primitive.getResult(1).getType()));
//                                    chisel_code += "\t" + memName + "_test_data := " + memName + ".test_data\n";
                                }
                            }
                        }
                    }
                }
//                chisel_code += "\t}\n"
//                               "\tval main = Module(new main)\n";

            }

            chisel_code += "}\n";
            chisel_code = "import chisel3._\n"
                          "import chisel3.util._\n"
                          "import chisel3.tester._\n"
                          "import chisel3.experimental.BundleLiterals\n"
                          "import utest._\n"
                          "import chisel3.experimental._\n"
                          "import hls._\n\n"
                          + chisel_code;
            std::cout << chisel_code << std::endl;
            exit(-1);
        }
    };

    std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
    createDumpChiselPass() {
        return std::make_unique<DumpChiselPass>();
    }

} // namespace mlir
