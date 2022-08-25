#include "HEC/HEC.h"

#include <iostream>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace hec;

//===----------------------------------------------------------------------===//
// DesignOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyDesignOp(DesignOp design) {
    // if (!design.getMainComponent())
    //     return design.emitOpError("must contain one \"main\" component");
    return success();
}

//===----------------------------------------------------------------------===//
// ComponentOp
//===----------------------------------------------------------------------===//

StateSetOp ComponentOp::getStateSetOp() {
    return *getBody()->getOps<StateSetOp>().begin();
}

GraphOp ComponentOp::getGraphOp() {
    return *getBody()->getOps<GraphOp>().begin();
}

// Returns the type of a given component as a function type.
static FunctionType getComponentType(ComponentOp component) {
    return component.getTypeAttr().getValue().cast<FunctionType>();
}

// Returns the port information for a given component
SmallVector<ComponentPortInfo> mlir::hec::getComponentPortInfo(Operation *op) {
    assert(isa<ComponentOp>(op) && "Can only get port info from a ComponentOp");
    auto component = dyn_cast<ComponentOp>(op);
    auto portTypes = getComponentType(component).getInputs();
    auto portNamesAttr = component.portNames();
    uint64_t numInPorts = component.numInPorts();

    SmallVector<ComponentPortInfo> results;
    for (uint64_t i = 0, e = portNamesAttr.size(); i != e; ++i) {
        auto dir = i < numInPorts ? PortDirection::INPUT : PortDirection::OUTPUT;
        results.push_back({portNamesAttr[i].cast<StringAttr>(), portTypes[i], dir});
    }
    return results;
}

static void printComponentOp(OpAsmPrinter &p, ComponentOp &op) {
    auto componentName =
            op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                    .getValue();
    p << "hec.component ";
    p.printSymbolName(componentName);

    auto ports = getComponentPortInfo(op);
    SmallVector<ComponentPortInfo, 4> inPorts, outPorts;
    for (auto &&port : ports) {
        if (port.direction == PortDirection::INPUT)
            inPorts.push_back(port);
        else
            outPorts.push_back(port);
    }

    // auto printPortList = [&](auto ports)
    // {
    //   p << "(";
    //   llvm::interleaveComma(ports, p, [&](auto port)
    //                         { p << "%" << port.name.getValue() << ": " <<
    //                         port.type; });
    //   p << ")";
    // };

    auto numInPorts = op.numInPorts();
    auto numPorts = op.getNumArguments();
    uint64_t count = 0;
    p << "(";
    if (op.getNumArguments() == 0) {
        p << ") -> ()";
    } else if (numInPorts == 0) {
        p << ") -> (";
    }

    for (auto arg : op.getArguments()) {
        p.printOperand(arg);
        p << ": " << arg.getType();

        count += 1;
        if (count == numInPorts)
            p << ") -> (";
        else if (count == numPorts)
            p << ")";
        else
            p << ",";
    }
    // printPortList(inPorts);
    // p << " -> ";
    // printPortList(outPorts);
    p << "\n\t\t{interface=\"" << op.interfc() << "\", style=\"" << op.style()
      << "\"}";

    p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
            /*printBlockTerminators=*/true,
            /*printEmptyBlock=*/false);

    p.printOptionalAttrDict(op->getAttrs(),
            /*elidedAttrs=*/{"interfc", "style", "numInPorts",
                             "portNames", "sym_name", "type"});
}

/// Parses the ports of a HEC component signature, and adds the corresponding
/// port names to `attrName`.
static ParseResult
parsePortDefList(OpAsmParser &parser, OperationState &result,
                 SmallVectorImpl<OpAsmParser::OperandType> &ports,
                 SmallVectorImpl<Type> &portTypes) {
    if (parser.parseLParen())
        return failure();

    do {
        OpAsmParser::OperandType port;
        Type portType;
        if (failed(parser.parseOptionalRegionArgument(port)) ||
            failed(parser.parseOptionalColon()) ||
            failed(parser.parseType(portType)))
            continue;
        ports.push_back(port);
        portTypes.push_back(portType);
    } while (succeeded(parser.parseOptionalComma()));

    return parser.parseRParen();
}

/// Parses the signature of a HEC component.
static ParseResult
parseComponentSignature(OpAsmParser &parser, OperationState &result,
                        SmallVectorImpl<OpAsmParser::OperandType> &ports,
                        SmallVectorImpl<Type> &portTypes) {
    if (parsePortDefList(parser, result, ports, portTypes))
        return failure();

    // Record the number of input ports.
    size_t numInPorts = ports.size();

    if (parser.parseArrow() || parsePortDefList(parser, result, ports, portTypes))
        return failure();

    auto *context = parser.getBuilder().getContext();
    // Add attribute for port names; these are currently
    // just inferred from the SSA names of the component.
    SmallVector<Attribute> portNames(ports.size());
    llvm::transform(ports, portNames.begin(), [&](auto port) -> StringAttr {
        StringRef name = port.name;
        if (name.startswith("%"))
            name = name.drop_front();
        return StringAttr::get(context, name);
    });
    result.addAttribute("portNames", ArrayAttr::get(context, portNames));

    // Record the number of input ports.
    result.addAttribute("numInPorts",
                        parser.getBuilder().getI64IntegerAttr(numInPorts));

    return success();
}

static ParseResult parseComponentOp(OpAsmParser &parser,
                                    OperationState &result) {
    // using namespace mlir::function_like_impl;

    StringAttr componentName;
    StringAttr wrappedOrNaked;
    StringAttr style;
    if (parser.parseSymbolName(componentName, SymbolTable::getSymbolAttrName(),
                               result.attributes))
        return failure();

    SmallVector<OpAsmParser::OperandType> ports;
    SmallVector<Type> portTypes;
    if (parseComponentSignature(parser, result, ports, portTypes))
        return failure();

    // Build the component's type for FunctionLike trait. All ports are listed as
    // arguments so they may be accessed within the component.
    auto type =
            parser.getBuilder().getFunctionType(portTypes, /*resultTypes=*/{});
    result.addAttribute(ComponentOp::getTypeAttrName(), TypeAttr::get(type));

    if (parser.parseLBrace() || parser.parseKeyword("interface") ||
        parser.parseEqual())
        return failure();
    if (parser.parseAttribute(wrappedOrNaked))
        return failure();
    if (parser.parseComma() || parser.parseKeyword("style") ||
        parser.parseEqual())
        return failure();
    if (parser.parseAttribute(style))
        return failure();
    if (parser.parseRBrace())
        return failure();

    result.addAttribute("interfc", wrappedOrNaked);
    result.addAttribute("style", style);

    auto *body = result.addRegion();
    if (parser.parseRegion(*body, ports, portTypes))
        return failure();

    if (body->empty())
        body->push_back(new Block());

    mlir::NamedAttrList additionalAttrs;
    if (!parser.parseOptionalAttrDict(additionalAttrs)) {
        for (auto attr : additionalAttrs) {
            result.addAttribute(attr.first, attr.second);
        }
    }

    return success();
}

static LogicalResult verifyComponentOp(ComponentOp op) {
    // Verify there is exactly one of either section:
    //    hec.graph, hec.stateset，hec.stageset
    // corresponding to style attribute
    /*
    uint32_t numStateSet = 0, numGraph = 0;
    for (auto &bodyOp : *op.getBody()) {
      if (isa<StateSetOp>(bodyOp))
        ++numStateSet;
      else if (isa<GraphOp>(bodyOp))
        ++numGraph;
    }
    llvm::StringRef style = op.style();
    if (numStateSet + numGraph != 1 || (style == "STG" && numStateSet != 1) ||
        (style == "handshake" && numGraph != 1))
      return op.emitOpError()
             << "hec.component must contain either a hec.stateset"
                " or a hec.graph according to style";
    */

    // Verify the number of input ports.
    SmallVector<ComponentPortInfo> componentPorts = getComponentPortInfo(op);
    uint64_t expectedNumInPorts =
            op->getAttrOfType<IntegerAttr>("numInPorts").getInt();
    uint64_t actualNumInPorts = llvm::count_if(componentPorts, [](auto port) {
        return port.direction == PortDirection::INPUT;
    });
    if (expectedNumInPorts != actualNumInPorts)
        return op.emitOpError()
                << "has mismatched number of in ports. Expected: "
                << expectedNumInPorts << ", actual: " << actualNumInPorts;

    // // Verify the component has the following ports.
    // // TODO(Calyx): Eventually, we want to attach attributes to these
    // arguments. bool go = false, clk = false, reset = false, done = false; for
    // (auto &&port : componentPorts)
    // {
    //   if (!port.type.isInteger(1))
    //     // Each of the ports has bit width 1.
    //     continue;

    //   StringRef portName = port.name.getValue();
    //   if (port.direction == PortDirection::OUTPUT)
    //   {
    //     done |= (portName == "done");
    //   }
    //   else
    //   {
    //     go |= (portName == "go");
    //     clk |= (portName == "clk");
    //     reset |= (portName == "reset");
    //   }
    //   if (go && clk && reset && done)
    //     return success();
    // }
    // return op->emitOpError() << "does not have required 1-bit input ports `go`,
    // "
    //                             "`clk`, `reset`, and output port `done`";

    return success();
}

void ComponentOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name, ArrayRef<ComponentPortInfo> ports,
                        StringAttr interfc, StringAttr style) {
    // using namespace mlir::function_like_impl;

    std::cerr << "Build a component" << std::endl;

    result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

    SmallVector<Type, 8> portTypes;
    SmallVector<Attribute, 8> portNames;
    uint64_t numInPorts = 0;

    for (auto port : ports) {
        if (port.direction == PortDirection::INPUT)
            ++numInPorts;
        portNames.push_back(port.name);
        portTypes.push_back(port.type);
        std::cerr << "{" << port.name.getValue().str() << ": ";

        std::string typeName;
        llvm::raw_string_ostream stro(typeName);
        port.type.print(stro);
        stro.flush();

        std::cerr << typeName;
        std::cerr << "}";
    }

    std::cerr << std::endl;

    // Build the function type of the component.
    auto functionType = builder.getFunctionType(portTypes, {});
    result.addAttribute(getTypeAttrName(), TypeAttr::get(functionType));

    // Record the port names and number of input ports of the component.
    result.addAttribute("portNames", builder.getArrayAttr(portNames));
    result.addAttribute("numInPorts", builder.getI64IntegerAttr(numInPorts));

    result.addAttribute("interfc", interfc);
    result.addAttribute("style", style);

    // Create a single-blocked region.
    result.addRegion();
    Region *regionBody = result.regions[0].get();
    Block *block = new Block();
    regionBody->push_back(block);

    // Add all ports to the body block.
    block->addArguments(portTypes);

    // Insert the WiresOp and ControlOp.
    IRRewriter::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(block);
    if (style.getValue() == "STG") {
        auto stateset = builder.create<StateSetOp>(result.location);
        stateset.getRegion().push_back(new mlir::Block);
    } else if (style.getValue() == "pipeline") {
        auto stageset = builder.create<StageSetOp>(result.location);
        stageset.getRegion().push_back(new mlir::Block);
    } else
        builder.create<GraphOp>(result.location);
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

ComponentOp InstanceOp::getReferencedComponent() {
    auto design = (*this)->getParentOfType<DesignOp>();
    if (!design)
        return nullptr;
    return design.lookupSymbol<ComponentOp>(componentName());
}

/// Provide meaningful names to the result values of a CellOp.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    auto component = getReferencedComponent();
    auto portNames = component.portNames();

    std::string prefix = instanceName().str() + ".";
    for (size_t i = 0, e = portNames.size(); i != e; ++i) {
        StringRef portName = portNames[i].cast<StringAttr>().getValue();
        setNameFn(getResult(i), prefix + portName.str());
    }
}

static LogicalResult verifyInstanceOp(InstanceOp instance) {
    if (instance.componentName() == "main")
        return instance.emitOpError("cannot reference the main component.");

    // Verify the referenced component exists in this program.
    ComponentOp referencedComponent = instance.getReferencedComponent();
    if (!referencedComponent)
        return instance.emitOpError()
                << "is referencing component: " << instance.componentName()
                << ", which does not exist.";

    // Verify the referenced component is not instantiating itself.
    auto parentComponent = instance->getParentOfType<ComponentOp>();
    if (parentComponent == referencedComponent)
        return instance.emitOpError()
                << "is a recursive instantiation of its parent component: "
                << instance.componentName();

    // Verify the instance result ports with those of its referenced component.
    SmallVector<ComponentPortInfo> componentPorts =
            getComponentPortInfo(referencedComponent);

    size_t numResults = instance.getNumResults();
    if (numResults != componentPorts.size())
        return instance.emitOpError()
                << "has a wrong number of results; expected: "
                << componentPorts.size() << " but got " << numResults;

    for (size_t i = 0; i != numResults; ++i) {
        auto resultType = instance.getResult(i).getType();
        auto expectedType = componentPorts[i].type;
        if (resultType == expectedType)
            continue;
        return instance.emitOpError()
                << "result type for " << componentPorts[i].name << " must be "
                << expectedType << ", but got " << resultType;
    }
    return success();
}

//===----------------------------------------------------------------------===//
// PrimitiveOp
//===----------------------------------------------------------------------===//

// Todo: Lookup the primitive component by name.
// Returns the port information for a given primitive

SmallVector<ComponentPortInfo> PrimitiveOp::getPrimitivePortInfo() {
    StringAttr name = primitiveNameAttr();
    SmallVector<ComponentPortInfo> results;

    if (name.getValue() == "register") {
        results.push_back({StringAttr::get((*this)->getContext(), "reg"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INOUT});
    } else if (name.getValue() == "add_integer" ||
               name.getValue() == "sub_integer" ||
               name.getValue() == "mul_integer" ||
               name.getValue() == "div_integer") {
        results.push_back({StringAttr::get((*this)->getContext(), "operand0"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "operand1"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "result"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "trunc_integer") {
        results.push_back({StringAttr::get((*this)->getContext(), "operand"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "result"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue().contains("cmp_integer")) {
        results.push_back({StringAttr::get((*this)->getContext(), "operand0"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "operand1"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "result"),
                           IntegerType::get((*this)->getContext(), 1),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "add_float" || name.getValue() == "sub_float" ||
               name.getValue() == "mul_float" || name.getValue() == "div_float") {
        results.push_back({StringAttr::get((*this)->getContext(), "operand0"),
                           FloatType::getF32((*this)->getContext()),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "operand1"),
                           FloatType::getF32((*this)->getContext()),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "result"),
                           FloatType::getF32((*this)->getContext()),
                           PortDirection::OUTPUT});
    } else if (name.getValue().contains("cmp_float")) {
        results.push_back({StringAttr::get((*this)->getContext(), "operand0"),
                           FloatType::getF32((*this)->getContext()),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "operand1"),
                           FloatType::getF32((*this)->getContext()),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "result"),
                           IntegerType::get((*this)->getContext(), 1),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "sitofp") {
        results.push_back({StringAttr::get((*this)->getContext(), "operand"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "result"),
                           FloatType::getF32((*this)->getContext()),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "fptosi") {
        results.push_back({StringAttr::get((*this)->getContext(), "operand"),
                           FloatType::getF32((*this)->getContext()),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "result"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue().contains("mem")) {
        auto rw = (*this)->getAttr("ports").cast<mlir::StringAttr>();
        assert(rw != nullptr && "Must provide read/write for mem");
        if (rw.getValue() == "r") {
            results.push_back({StringAttr::get((*this)->getContext(), "r_en"),
                               getType(0), PortDirection::INPUT});
            results.push_back({StringAttr::get((*this)->getContext(), "addr"),
                               getType(1), PortDirection::INPUT});
            results.push_back({StringAttr::get((*this)->getContext(), "r_data"),
                               getType(2), PortDirection::OUTPUT});
        } else if (rw.getValue() == "w") {
            results.push_back({StringAttr::get((*this)->getContext(), "w_en"),
                               getType(0), PortDirection::INPUT});
            // FIXME: Need to cope with r_en signal
            //            results.push_back({StringAttr::get((*this)->getContext(),
            //            "r_en"),
            //                               getType(1), PortDirection::INPUT});
            results.push_back({StringAttr::get((*this)->getContext(), "addr"),
                               getType(2), PortDirection::INPUT});
            results.push_back({StringAttr::get((*this)->getContext(), "w_data"),
                               getType(3), PortDirection::INPUT});
        } else if (rw.getValue() == "rw") {
            results.push_back({StringAttr::get((*this)->getContext(), "w_en"),
                               getType(0), PortDirection::INPUT});
            results.push_back({StringAttr::get((*this)->getContext(), "r_en"),
                               getType(1), PortDirection::INPUT});
            results.push_back({StringAttr::get((*this)->getContext(), "addr"),
                               getType(2), PortDirection::INPUT});
            results.push_back({StringAttr::get((*this)->getContext(), "w_data"),
                               getType(3), PortDirection::INPUT});
            results.push_back({StringAttr::get((*this)->getContext(), "r_data"),
                               getType(4), PortDirection::OUTPUT});
        }
    } else if (name.getValue() == "buffer") {
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataOut"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "merge") {
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn.0"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn.1"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataOut"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "branch") {
        results.push_back({StringAttr::get((*this)->getContext(), "condition"),
                           IntegerType::get((*this)->getContext(), 1),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataOut.0"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataOut.1"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue().contains("load")) {
        results.push_back({StringAttr::get((*this)->getContext(), "address_in"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "data_out"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "address_out"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "data_in"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "control"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
    } else if (name.getValue().contains("store")) {
        results.push_back({StringAttr::get((*this)->getContext(), "address_in"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "data_in"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "address_out"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "data_out"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "control"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
    } else if (name.getValue().contains("fork")) {
        if (name.getValue().contains(":")) {
            results.push_back({StringAttr::get((*this)->getContext(), "dataIn"),
                               IntegerType::get((*this)->getContext(), 32),
                               PortDirection::INPUT});
            std::string forkName = name.getValue().str();
            unsigned num = std::atoi(forkName.substr(forkName.find(":") + 1).c_str());
            for (unsigned idx = 0; idx < num; ++idx) {
                results.push_back({StringAttr::get((*this)->getContext(),
                                                   "dataOut." + std::to_string(idx)),
                                   IntegerType::get((*this)->getContext(), 32),
                                   PortDirection::OUTPUT});
            }
        } else {
            results.push_back({StringAttr::get((*this)->getContext(), "dataIn"),
                               IntegerType::get((*this)->getContext(), 32),
                               PortDirection::INPUT});
            results.push_back({StringAttr::get((*this)->getContext(), "dataOut.0"),
                               IntegerType::get((*this)->getContext(), 32),
                               PortDirection::OUTPUT});
            results.push_back({StringAttr::get((*this)->getContext(), "dataOut.1"),
                               IntegerType::get((*this)->getContext(), 32),
                               PortDirection::OUTPUT});
        }
    } else if (name.getValue().contains("fifo")) {
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataOut"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
    } else if (name.getValue().contains("dyn_Mem")) {
        if (name.getValue().contains(":")) {
            std::string memName = name.getValue().str();
            unsigned loadnum =
                    std::atoi(memName
                                      .substr(memName.find(":") + 1,
                                              memName.find(",") - memName.find(":") - 1)
                                      .c_str());
            unsigned storenum =
                    std::atoi(memName.substr(memName.find(",") + 1).c_str());
            for (unsigned idx = 0; idx < loadnum; ++idx) {
                results.push_back(
                        {StringAttr::get((*this)->getContext(),
                                         "load_address." + std::to_string(idx)),
                         IntegerType::get((*this)->getContext(), 32),
                         PortDirection::INPUT});
                results.push_back({StringAttr::get((*this)->getContext(),
                                                   "load_data." + std::to_string(idx)),
                                   IntegerType::get((*this)->getContext(), 32),
                                   PortDirection::OUTPUT});
            }
            for (unsigned idx = 0; idx < storenum; ++idx) {
                results.push_back(
                        {StringAttr::get((*this)->getContext(),
                                         "store_address." + std::to_string(idx)),
                         IntegerType::get((*this)->getContext(), 32),
                         PortDirection::INPUT});
                results.push_back({StringAttr::get((*this)->getContext(),
                                                   "store_data." + std::to_string(idx)),
                                   IntegerType::get((*this)->getContext(), 32),
                                   PortDirection::INPUT});
            }
        }
    } else if (name.getValue() == "control_merge") {
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn.0"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn.1"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataOut"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "condition"),
                           IntegerType::get((*this)->getContext(), 1),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "shift_left") {
        results.push_back({StringAttr::get((*this)->getContext(), "operand0"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "operand1"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "result"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "select") {
        results.push_back({StringAttr::get((*this)->getContext(), "condition"),
                           IntegerType::get((*this)->getContext(), 1),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn.0"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn.1"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataOut"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "constant") {
        results.push_back({StringAttr::get((*this)->getContext(), "control"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataOut"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "mux_dynamic") {
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn.0"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn.1"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataOut"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "condition"),
                           IntegerType::get((*this)->getContext(), 1),
                           PortDirection::INPUT});
    } else if (name.getValue() == "and") {
        results.push_back({StringAttr::get((*this)->getContext(), "operand0"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "operand1"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "result"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "fptosi") {
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataOut"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "neg_float") {
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get((*this)->getContext(), "dataOut"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "sink") {
        results.push_back({StringAttr::get((*this)->getContext(), "dataIn"),
                           IntegerType::get((*this)->getContext(), 32),
                           PortDirection::INPUT});
    } else {
        std::cerr << name.getValue().str() << std::endl;
        assert(0 && "hec.primitive op has an undefined primitiveName");
    }
    return results;
}

/// Provide meaningful names to the result values of a PrimitiveOp.
void PrimitiveOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    auto portInfos = getPrimitivePortInfo();

    std::string prefix = instanceName().str() + ".";

    assert(portInfos.size() == getResults().size() &&
           "# of results must meet the primitive");
    for (size_t i = 0, e = portInfos.size(); i != e; ++i) {
        StringRef portName = portInfos[i].name.getValue();
        setNameFn(getResult(i), prefix + portName.str());
    }
}

static LogicalResult verifyPrimitiveOp(PrimitiveOp primitive) {
    // Verify the referenced primitive component exists.
    // ComponentOp referencedComponent = instance.getReferencedComponent();
    // if (!referencedComponent)
    //   return instance.emitOpError()
    //          << "is referencing component: " << instance.componentName()
    //          << ", which does not exist.";

    // Verify the instance result ports with those of its referenced component.
    // SmallVector<ComponentPortInfo> componentPorts =
    //     getComponentPortInfo(referencedComponent);

    // size_t numResults = instance.getNumResults();
    // if (numResults != componentPorts.size())
    //   return instance.emitOpError()
    //          << "has a wrong number of results; expected: "
    //          << componentPorts.size() << " but got " << numResults;

    // for (size_t i = 0; i != numResults; ++i)
    // {
    //   auto resultType = instance.getResult(i).getType();
    //   auto expectedType = componentPorts[i].type;
    //   if (resultType == expectedType)
    //     continue;
    //   return instance.emitOpError()
    //          << "result type for " << componentPorts[i].name << " must be "
    //          << expectedType << ", but got " << resultType;
    // }
    return success();
}

//===----------------------------------------------------------------------===//
// StateSetOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyStateSetOp(StateSetOp stateset) {
    // auto component = stateset->getParentOfType<ComponentOp>();

    // TODO: check states and transitions

    return success();
}

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//
static void printStateOp(OpAsmPrinter &p, StateOp &op) {
    auto stateName =
            op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                    .getValue();
    p << "hec.state ";
    p.printSymbolName(stateName);

    if (op.initial())
        p << "*";

    p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
            /*printBlockTerminators=*/true,
            /*printEmptyBlock=*/false);
}

static ParseResult parseStateOp(OpAsmParser &parser, OperationState &result) {
    // using namespace mlir::function_like_impl;

    StringAttr stateName;
    IntegerAttr initial;
    if (parser.parseSymbolName(stateName, SymbolTable::getSymbolAttrName(),
                               result.attributes))
        return failure();

    if (parser.parseOptionalStar()) {
        initial =
                parser.getBuilder().getIntegerAttr(parser.getBuilder().getI1Type(), 0);
    } else {
        initial =
                parser.getBuilder().getIntegerAttr(parser.getBuilder().getI1Type(), 1);
    }
    result.addAttribute("initial", initial);

    auto *body = result.addRegion();
    if (parser.parseRegion(*body))
        return failure();

    if (body->empty())
        body->push_back(new Block());
    return success();
}

static LogicalResult verifyStateOp(StateOp op) {
    // TODO: Verify there exists a initial state
    return success();
}

void StateOp::build(OpBuilder &builder, OperationState &result, StringAttr name,
                    IntegerAttr initial) {
    // using namespace mlir::function_like_impl;

    result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

    result.addAttribute("initial", initial);

    // Create a single-blocked region.
    result.addRegion();
    Region *regionBody = result.regions[0].get();
    Block *block = new Block();
    regionBody->push_back(block);
}

//===----------------------------------------------------------------------===//
// TransitionOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyTransitionOp(TransitionOp transition) {
    // TODO: check the if-elif-elif-...-else style.
    return success();
}

//===----------------------------------------------------------------------===//
// GotoOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyGotoOp(GotoOp gotoop) {
    // TODO: check the dest is in the stateset
    return success();
}

//===----------------------------------------------------------------------===//
// CDoneOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyCDoneOp(CDoneOp done) {
    // TODO: arguments must be consistent with component's out-ports
    return success();
}

//===----------------------------------------------------------------------===//
// DoneOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyDoneOp(DoneOp done) {
    // TODO: arguments must be consistent with component's out-ports
    return success();
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyGraphOp(GraphOp graph) {
    // TODO: check handshake linkage behaviors
    return success();
}

//===----------------------------------------------------------------------===//
// StageSetOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyStageSetOp(StageSetOp stageset) { return success(); }

//===----------------------------------------------------------------------===//
// StageOp
//===----------------------------------------------------------------------===//

static void printStageOp(OpAsmPrinter &p, StageOp &op) {
    auto stageName =
            op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                    .getValue();
    p << "hec.stage ";
    p.printSymbolName(stageName);

    p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
            /*printBlockTerminators=*/true,
            /*printEmptyBlock=*/false);
}

static ParseResult parseStageOp(OpAsmParser &parser, OperationState &result) {
    // using namespace mlir::function_like_impl;

    StringAttr stageName;
    if (parser.parseSymbolName(stageName, SymbolTable::getSymbolAttrName(),
                               result.attributes))
        return failure();

    auto *body = result.addRegion();
    if (parser.parseRegion(*body))
        return failure();

    if (body->empty())
        body->push_back(new Block());
    return success();
}

static LogicalResult verifyStageOp(StageOp stage) { return success(); }

void StageOp::build(OpBuilder &builder, OperationState &result,
                    StringAttr name) {
    // using namespace mlir::function_like_impl;

    result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);
    // Create a single-blocked region.
    result.addRegion();
    Region *regionBody = result.regions[0].get();
    Block *block = new Block();
    regionBody->push_back(block);
}

#define GET_OP_CLASSES

#include "HEC/HEC.cpp.inc"
