#include "HEC/HEC.h"

#include <iostream>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
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
LogicalResult DesignOp::verify() {
    if (!this->getMainComponent())
        return this->emitOpError("must contain one \"main\" component");
    return success();
}

//===----------------------------------------------------------------------===//
// ComponentOp
//===----------------------------------------------------------------------===//

StateSetOp ComponentOp::getStateSetOp() {
    return *(getBody().getOps<StateSetOp>().begin());
}

GraphOp ComponentOp::getGraphOp() {
    return *(getBody().getOps<GraphOp>().begin());
}

// Returns the type of a given component as a function type.
// static FunctionType getComponentType(ComponentOp component) {
//     return component.getTypeAttr().getValue().cast<FunctionType>();
// }

// Returns the port information for a given component
SmallVector<ComponentPortInfo> mlir::hec::getComponentPortInfo(Operation *op) {
    assert(isa<ComponentOp>(op) && "Can only get port info from a ComponentOp");
    auto component = dyn_cast<ComponentOp>(op);
    auto portTypes = component.getArgumentTypes();
    auto portNamesAttr = component.getPortNames();
    uint64_t numInPorts = component.getNumInPorts();

    SmallVector<ComponentPortInfo> results;
    for (uint64_t i = 0, e = portNamesAttr.size(); i != e; ++i) {
        auto dir = i < numInPorts ? PortDirection::INPUT : PortDirection::OUTPUT;
        results.push_back({portNamesAttr[i].cast<StringAttr>(), portTypes[i], dir});
    }
    return results;
}

void ComponentOp::print(OpAsmPrinter &p) {
    auto componentName = (*this)->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
        .getValue();

    // p << "hec.component ";
    p << " ";
    p.printSymbolName(componentName);

    auto ports = getComponentPortInfo(*this);
    SmallVector<ComponentPortInfo, 4> inPorts, outPorts;
    for (auto &&port : ports) {
        if (port.direction == PortDirection::INPUT)
            inPorts.push_back(port);
        else
            outPorts.push_back(port);
    }

    auto numInPorts = this->getNumInPorts();
    auto numPorts = this->getNumArguments();
    uint64_t count = 0;
    p << "(";
    if (this->getNumArguments() == 0) {
        p << ") -> ()";
    } else if (numInPorts == 0) {
        p << ") -> (";
    }

    for (auto arg : this->getArguments()) {
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
    
    p << "\n\t\t{interface=\"" << this->getInterfc() << "\", style=\"" << this->getStyle()
      << "\"}";

    p.printRegion(this->getBody(), /*printEntryBlockArgs=*/false,
            /*printBlockTerminators=*/true,
            /*printEmptyBlock=*/false);

    p.printOptionalAttrDict((*this)->getAttrs(),
            /*elidedAttrs=*/{"interfc", "style", "numInPorts",
                             "portNames", "sym_name", "function_type"});
}

// Parses the ports of a HEC component signature, and adds the corresponding
// port names to `attrName`.
ParseResult parsePortDefList(OpAsmParser &parser, OperationState &result,
                 SmallVectorImpl<OpAsmParser::Argument> &ports) {

    if (parser.parseLParen())
        return failure();
    do {
        OpAsmParser::Argument port;
        Type portType;
        if (parser.parseOptionalArgument(port, true).value())
            continue;

        ports.push_back(port);
    } while (succeeded(parser.parseOptionalComma()));

    return parser.parseRParen();
}

/// Parses the signature of a HEC component.
ParseResult parseComponentSignature(OpAsmParser &parser, OperationState &result,
                        SmallVectorImpl<OpAsmParser::Argument> &ports) {
    if (parsePortDefList(parser, result, ports))
        return failure();
    // Record the number of input ports.
    size_t numInPorts = ports.size();

    if (parser.parseArrow() || parsePortDefList(parser, result, ports))
        return failure();

    auto *context = parser.getBuilder().getContext();
    // Add attribute for port names; these are currently
    // just inferred from the SSA names of the component.
    SmallVector<Attribute> portNames(ports.size());
    llvm::transform(ports, portNames.begin(), [&](auto port) -> StringAttr {
        StringRef name = port.ssaName.name;
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

ParseResult ComponentOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
    // using namespace mlir::function_like_impl;

    StringAttr componentName;
    StringAttr wrappedOrNaked;
    StringAttr style;
    if (parser.parseSymbolName(componentName, SymbolTable::getSymbolAttrName(),
                               result.attributes))
        return failure();

    SmallVector<OpAsmParser::Argument> ports;
    if (parseComponentSignature(parser, result, ports))
        return failure();

    // Build the component's type for FunctionLike trait. All ports are listed as
    // arguments so they may be accessed within the component.
    SmallVector<Type> portTypes;
    for (auto port : ports) {
        portTypes.push_back(port.type);
    }
    
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
    if (parser.parseRegion(*body, ports))
        return failure();

    if (body->empty())
        body->push_back(new Block());

    mlir::NamedAttrList additionalAttrs;
    if (!parser.parseOptionalAttrDict(additionalAttrs)) {
        for (auto attr : additionalAttrs) {
            result.addAttribute(attr.getName(), attr.getValue());
        }
    }

    return success();
}

LogicalResult ComponentOp::verify() {
    // Verify the number of input ports.
    SmallVector<ComponentPortInfo> componentPorts = getComponentPortInfo(*this);
    uint64_t expectedNumInPorts =
            (*this)->getAttrOfType<IntegerAttr>("numInPorts").getInt();
    uint64_t actualNumInPorts = llvm::count_if(componentPorts, [](auto port) {
        return port.direction == PortDirection::INPUT;
    });
    if (expectedNumInPorts != actualNumInPorts)
        return this->emitOpError()
                << "has mismatched number of in ports. Expected: "
                << expectedNumInPorts << ", actual: " << actualNumInPorts;
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
    SmallVector<Location, 8> locations(portTypes.size(), result.location);
    // for (unsigned i = 0; i < portTypes.size(); ++i) {
    //     locations.push_back(result.location);
    // }
    block->addArguments(portTypes, locations);

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
    return design.lookupSymbol<ComponentOp>(getComponentName());
}

// Provide meaningful names to the result values of a CellOp.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    auto component = getReferencedComponent();
    auto portNames = component.getPortNames();

    std::string prefix = getInstanceName().str() + ".";
    for (size_t i = 0, e = portNames.size(); i != e; ++i) {
        StringRef portName = portNames[i].cast<StringAttr>().getValue();
        setNameFn(getResult(i), prefix + portName.str());
    }
}

LogicalResult InstanceOp::verify() {
    if (this->getComponentName() == "main")
        return this->emitOpError("cannot reference the main component.");

    // Verify the referenced component exists in this program.
    ComponentOp referencedComponent = this->getReferencedComponent();
    if (!referencedComponent)
        return this->emitOpError()
                << "is referencing component: " << this->getComponentName()
                << ", which does not exist.";

    // Verify the referenced component is not instantiating itself.
    auto parentComponent = (*this)->getParentOfType<ComponentOp>();
    if (parentComponent == referencedComponent)
        return this->emitOpError()
                << "is a recursive instantiation of its parent component: "
                << this->getComponentName();

    // Verify the instance result ports with those of its referenced component.
    SmallVector<ComponentPortInfo> componentPorts =
            getComponentPortInfo(referencedComponent);

    size_t numResults = this->getNumResults();
    if (numResults != componentPorts.size())
        return this->emitOpError()
                << "has a wrong number of results; expected: "
                << componentPorts.size() << " but got " << numResults;

    for (size_t i = 0; i != numResults; ++i) {
        auto resultType = this->getResult(i).getType();
        auto expectedType = componentPorts[i].type;
        if (resultType == expectedType)
            continue;
        return this->emitOpError()
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
    StringAttr name = getPrimitiveNameAttr();
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

// Provide meaningful names to the result values of a PrimitiveOp.
void PrimitiveOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    auto portInfos = getPrimitivePortInfo();

    std::string prefix = getInstanceName().str() + ".";

    assert(portInfos.size() == getResults().size() &&
           "# of results must meet the primitive");
    for (size_t i = 0, e = portInfos.size(); i != e; ++i) {
        StringRef portName = portInfos[i].name.getValue();
        setNameFn(getResult(i), prefix + portName.str());
    }
}

// LogicalResult PrimitiveOp::verify(PrimitiveOp primitive) {
//     // Verify the referenced primitive component exists.
//     // ComponentOp referencedComponent = instance.getReferencedComponent();
//     // if (!referencedComponent)
//     //   return instance.emitOpError()
//     //          << "is referencing component: " << instance.componentName()
//     //          << ", which does not exist.";

//     // Verify the instance result ports with those of its referenced component.
//     // SmallVector<ComponentPortInfo> componentPorts =
//     //     getComponentPortInfo(referencedComponent);

//     // size_t numResults = instance.getNumResults();
//     // if (numResults != componentPorts.size())
//     //   return instance.emitOpError()
//     //          << "has a wrong number of results; expected: "
//     //          << componentPorts.size() << " but got " << numResults;

//     // for (size_t i = 0; i != numResults; ++i)
//     // {
//     //   auto resultType = instance.getResult(i).getType();
//     //   auto expectedType = componentPorts[i].type;
//     //   if (resultType == expectedType)
//     //     continue;
//     //   return instance.emitOpError()
//     //          << "result type for " << componentPorts[i].name << " must be "
//     //          << expectedType << ", but got " << resultType;
//     // }
//     return success();
// }

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//
void StateOp::print(OpAsmPrinter &p) {
    auto stateName =
            (*this)->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                    .getValue();
    // p << "hec.state ";
    p << " ";
    p.printSymbolName(stateName);

    if (this->getInitial())
        p << "*";

    p.printRegion(this->getBody(), /*printEntryBlockArgs=*/false,
            /*printBlockTerminators=*/true,
            /*printEmptyBlock=*/false);
}

ParseResult StateOp::parse(OpAsmParser &parser, OperationState &result) {
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

// //===----------------------------------------------------------------------===//
// // StageOp
// //===----------------------------------------------------------------------===//

void StageOp::print(OpAsmPrinter &p) {
    auto stageName =
            (*this)->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                    .getValue();
    // p << "hec.stage ";
    p << " ";
    p.printSymbolName(stageName);

    p.printRegion(this->getBody(), /*printEntryBlockArgs=*/false,
            /*printBlockTerminators=*/true,
            /*printEmptyBlock=*/false);
}

ParseResult StageOp::parse(OpAsmParser &parser, OperationState &result) {
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
