#include "TOR/TORTypes.h"
#include "TOR/TORDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <iostream>

#define GET_TYPEDEF_CLASSES
#include "TOR/TORTypes.cpp.inc"

namespace mlir
{
  namespace tor
  {
    //===----------------------------------------------------------------------===//
    // MemRefType
    //===----------------------------------------------------------------------===//

    /// This is a builder type that keeps local references to arguments. Arguments
    /// that are passed into the builder must out-live the builder.
    class MemRefType::Builder
    {
    public:
      // Build from another MemRefType.
      explicit Builder(MemRefType other)
          : shape(other.getShape()), elementType(other.getElementType()),
            /*affineMaps(other.getAffineMaps()),*/
            property(other.getProperty()),
            rw(other.getRw()) /*,
        memorySpace(other.getMemorySpace())*/
      {
      }

      // Build from scratch.
      Builder(llvm::ArrayRef<int64_t> shape, mlir::Type elementType)
          : shape(shape), elementType(elementType) /*, affineMaps()*/,
            property(), rw() {}

      Builder &setShape(ArrayRef<int64_t> newShape)
      {
        shape = newShape;
        return *this;
      }

      Builder &setElementType(Type newElementType)
      {
        elementType = newElementType;
        return *this;
      }

      // Builder &setAffineMaps(ArrayRef<AffineMap> newAffineMaps) {
      //   affineMaps = newAffineMaps;
      //   return *this;
      // }

      Builder &setProperty(llvm::ArrayRef<mlir::StringAttr> newProperty)
      {
        property = newProperty;
        return *this;
      }

      Builder &setRw(StringAttr newRw)
      {
        rw = newRw;
        return *this;
      }
      /*
  Builder &setMemorySpace(Attribute newMemorySpace) {
    memorySpace = newMemorySpace;
    return *this;
  }
*/

      operator MemRefType()
      {
        return MemRefType::get(shape, elementType, /*affineMaps,*/
                               property, rw /*, memorySpace*/);
      }

    private:
      // ArrayRef<AffineMap> affineMaps;

      ::llvm::ArrayRef<int64_t> shape;
      mlir::Type elementType;
      ::llvm::ArrayRef<StringAttr> property;
      mlir::StringAttr rw;
      // mlir::Attribute memorySpace;
    };

    inline bool BaseMemRefType::classof(Type type)
    {
      return type.isa<mlir::tor::MemRefType>();
    }
  } // end tor
} // end mlir

namespace mlir
{
  namespace tor
  {

    //===----------------------------------------------------------------------===//
    // ShapedType
    //===----------------------------------------------------------------------===//
    constexpr int64_t ShapedType::kDynamicSize;
    constexpr int64_t ShapedType::kDynamicStrideOrOffset;

    ShapedType ShapedType::clone(ArrayRef<int64_t> shape, Type elementType)
    {
      if (auto other = dyn_cast<MemRefType>())
      {
        MemRefType::Builder b(other);
        b.setShape(shape);
        b.setElementType(elementType);
        return b;
      }
      llvm_unreachable("Unhandled ShapedType clone case");
    }

    ShapedType ShapedType::clone(ArrayRef<int64_t> shape)
    {
      if (auto other = dyn_cast<MemRefType>())
      {
        MemRefType::Builder b(other);
        b.setShape(shape);
        return b;
      }
      llvm_unreachable("Unhandled ShapedType clone case");
    }

    ShapedType ShapedType::clone(Type elementType)
    {
      if (auto other = dyn_cast<MemRefType>())
      {
        MemRefType::Builder b(other);
        b.setElementType(elementType);
        return b;
      }
      llvm_unreachable("Unhandled ShapedType clone hit");
    }

    mlir::Type ShapedType::getElementType() const
    {
      return TypeSwitch<Type, Type>(*this)
          .Case<VectorType, RankedTensorType, UnrankedTensorType, MemRefType,
                UnrankedMemRefType>([](auto ty)
                                    { return ty.getElementType(); });
    }

    unsigned ShapedType::getElementTypeBitWidth() const
    {
      return getElementType().getIntOrFloatBitWidth();
    }

    int64_t ShapedType::getNumElements() const
    {
      assert(hasStaticShape() && "cannot get element count of dynamic shaped type");
      auto shape = getShape();
      int64_t num = 1;
      for (auto dim : shape)
      {
        num *= dim;
        assert(num >= 0 && "integer overflow in element count computation");
      }
      return num;
    }

    int64_t ShapedType::getRank() const
    {
      assert(hasRank() && "cannot query rank of unranked shaped type");
      return getShape().size();
    }

    bool ShapedType::hasRank() const
    {
      return true;
      // return !isa<UnrankedMemRefType, UnrankedTensorType>();
    }

    int64_t ShapedType::getDimSize(unsigned idx) const
    {
      assert(idx < getRank() && "invalid index for shaped type");
      return getShape()[idx];
    }

    bool ShapedType::isDynamicDim(unsigned idx) const
    {
      assert(false && "doesn't support dynamic dim");
      assert(idx < getRank() && "invalid index for shaped type");
      return isDynamic(getShape()[idx]);
    }

    unsigned ShapedType::getDynamicDimIndex(unsigned index) const
    {
      assert(false && "doesn't support dynamic dim");
      assert(index < getRank() && "invalid index");
      assert(ShapedType::isDynamic(getDimSize(index)) && "invalid index");
      return llvm::count_if(getShape().take_front(index), ShapedType::isDynamic);
    }

    /// Get the number of bits require to store a value of the given shaped type.
    /// Compute the value recursively since tensors are allowed to have vectors as
    /// elements.
    int64_t ShapedType::getSizeInBits() const
    {
      assert(false && "doesn't support getSizeInBits");
      return 0;
      // assert(hasStaticShape() &&
      //        "cannot get the bit size of an aggregate with a dynamic shape");

      // auto elementType = getElementType();
      // if (elementType.isIntOrFloat())
      //   return elementType.getIntOrFloatBitWidth() * getNumElements();

      // if (auto complexType = elementType.dyn_cast<ComplexType>()) {
      //   elementType = complexType.getElementType();
      //   return elementType.getIntOrFloatBitWidth() * getNumElements() * 2;
      // }

      // // Tensors can have vectors and other tensors as elements, other shaped types
      // // cannot.
      // assert(isa<TensorType>() && "unsupported element type");
      // assert((elementType.isa<VectorType, TensorType>()) &&
      //        "unsupported tensor element type");
      // return getNumElements() * elementType.cast<ShapedType>().getSizeInBits();
    }

    mlir::ArrayRef<int64_t> ShapedType::getShape() const
    {
      // if (auto vectorType = dyn_cast<VectorType>())
      //   return vectorType.getShape();
      // if (auto tensorType = dyn_cast<RankedTensorType>())
      //   return tensorType.getShape();
      return cast<MemRefType>().getShape();
    }

    int64_t ShapedType::getNumDynamicDims() const
    {
      assert(false && "doen't support dynamic dims");
      return 0;
      // return llvm::count_if(getShape(), isDynamic);
    }

    bool ShapedType::hasStaticShape() const
    {
      return true;
      // return hasRank() && llvm::none_of(getShape(), isDynamic);
    }

    bool ShapedType::hasStaticShape(ArrayRef<int64_t> shape) const
    {
      return hasStaticShape() && getShape() == shape;
    }

  } // end tor
} // end mlir

//===----------------------------------------------------------------------===//
// BaseMemRefType
//===----------------------------------------------------------------------===//

namespace mlir
{
  namespace tor
  {
    // Attribute BaseMemRefType::getMemorySpace() const {
    //   if (auto rankedMemRefTy = dyn_cast<MemRefType>())
    //     return rankedMemRefTy.getMemorySpace();
    //   return cast<UnrankedMemRefType>().getMemorySpace();
    // }
  } // end tor
} // end mlir

//===----------------------------------------------------------------------===//
// MemRefType
//===----------------------------------------------------------------------===//

//
namespace mlir
{
  namespace tor
  {

    ::mlir::Type MemRefType::parse(::mlir::MLIRContext *context,
                                   ::mlir::DialectAsmParser &parser)
    {
      SmallVector<int64_t, 2> dims;
      SmallVector<mlir::StringAttr, 2> properties;
      Type type;
      StringAttr rw;
      // Attribute memorySpace;

      if (parser.parseLess() || parser.parseDimensionList(dims, false) ||
          parser.parseType(type) || parser.parseComma())
        return Type();
      if (dims.size() == 0)
        return Type();
      if (parser.parseLSquare())
        return Type();

      // std::cerr << "get <d*t，[" << std::endl;
      while (parser.parseOptionalRSquare())
      {
        // std::cerr << "need one StringAttr" << std::endl;
        mlir::StringAttr strAttr;
        if (parser.parseAttribute(strAttr))
          return Type();
        properties.push_back(strAttr);
        // std::cerr << "get one StringAttr" << std::endl;
      }
      // std::cerr << "get ]" << std::endl;
      if (properties.size() > dims.size())
      {
        parser.emitError(parser.getNameLoc(), "property's size > dim's size");
        return Type();
      }

      if (parser.parseComma())
        return Type();

      // std::cerr << "get, " << std::endl;
      // std::cerr << "need one StringAttr" << std::endl;
      if (parser.parseAttribute(rw))
        return Type();
      // std::cerr << "get on StringAttr" << std::endl;

      // if (parser.parseAttribute(memorySpace))
      //   return Type();

      if (parser.parseGreater())
        return Type();

      // std::cerr << "get >" << std::endl;

      return MemRefType::getChecked(
          [&]() -> InFlightDiagnostic
          {
            return parser.emitError(parser.getCurrentLocation());
          },
          dims, type, properties, rw /*, memorySpace*/);
    }

    ::mlir::LogicalResult MemRefType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
        ::llvm::ArrayRef<int64_t> shape, Type elementType,
        ::llvm::ArrayRef<StringAttr> property,
        StringAttr rw /*, Attribute memorySpace*/)
    {

      if (shape.size() == 0 || property.size() > shape.size())
        return emitError() << "Wrong size";
      return success();
    }

    void MemRefType::print(::mlir::DialectAsmPrinter &printer) const
    {
      printer << "memref<";
      auto shape = getShape();
      printer << shape[0];
      for (auto dim : shape.drop_front())
      {
        printer << "x" << dim;
      }
      printer << "x";
      printer.printType(getElementType());
      printer << ", ";
      auto property = getProperty();
      printer << "[";
      if (!property.empty())
      {
        printer << "\"" << property[0].getValue() << "\"";
        for (auto i : property.drop_front())
          printer << ", \"" << i.getValue() << "\"";
      }
      printer << "], \"" << getRw().getValue() << "\""
              << ">";
    }

  } // end tor
} // end mlir

namespace mlir
{
  namespace tor
  {

    void TORDialect::registerTypes()
    {
      addTypes<
#define GET_TYPEDEF_LIST
#include "TOR/TORTypes.cpp.inc"
          >();
    }

    /// Parses a type registered to this dialect. Parse out the mnemonic then invoke
    /// the tblgen'd type parser dispatcher.
    Type TORDialect::parseType(DialectAsmParser &parser) const
    {
      llvm::StringRef mnemonic;
      if (parser.parseKeyword(&mnemonic))
        return Type();
      Type type;
      auto parseResult = generatedTypeParser(getContext(), parser, mnemonic, type);
      if (parseResult.hasValue())
        return type;
      return Type();
    }

    /// Print a type registered to this dialect. Try the tblgen'd type printer
    /// dispatcher then fail since all RTL types are defined via ODS.
    void TORDialect::printType(Type type, DialectAsmPrinter &printer) const
    {
      if (succeeded(generatedTypePrinter(type, printer)))
        return;
      llvm_unreachable("unexpected 'tor' type");
    }

    namespace detail
    {

      // mlir::Attribute skipDefaultMemorySpace(mlir::Attribute memorySpace) {
      //   mlir::IntegerAttr intMemorySpace =
      //       memorySpace.dyn_cast_or_null<mlir::IntegerAttr>();
      //   if (intMemorySpace && intMemorySpace.getValue() == 0)
      //     return nullptr;

      //   return memorySpace;
      // }
    } // end detail

  }
}