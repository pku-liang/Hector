#ifndef TOR_TYPES_H
#define TOR_TYPES_H

#include "TOR/TORDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

namespace mlir
{
  namespace tor
  {
    namespace detail
    {
      // mlir::Attribute skipDefaultMemorySpace(mlir::Attribute memorySpace);
    } // end detail

    /// This is a common base class between Vector, UnrankedTensor, RankedTensor,
    /// and MemRef types because they share behavior and semantics around shape,
    /// rank, and fixed element type. Any type with these semantics should inherit
    /// from ShapedType.
    class ShapedType : public Type
    {
    public:
      using Type::Type;

      static constexpr int64_t kDynamicSize = -1;
      static constexpr int64_t kDynamicStrideOrOffset =
          std::numeric_limits<int64_t>::min();

      /// Return clone of this type with new shape and element type.
      ShapedType clone(ArrayRef<int64_t> shape, Type elementType);
      ShapedType clone(ArrayRef<int64_t> shape);
      ShapedType clone(Type elementType);

      /// Return the element type.
      Type getElementType() const;

      /// If an element type is an integer or a float, return its width. Otherwise,
      /// abort.
      unsigned getElementTypeBitWidth() const;

      /// If it has static shape, return the number of elements. Otherwise, abort.
      int64_t getNumElements() const;

      /// If this is a ranked type, return the rank. Otherwise, abort.
      int64_t getRank() const;

      /// Whether or not this is a ranked type. Memrefs, vectors and ranked tensors
      /// have a rank, while unranked tensors do not.
      bool hasRank() const;

      /// If this is a ranked type, return the shape. Otherwise, abort.
      ArrayRef<int64_t> getShape() const;

      /// If this is unranked type or any dimension has unknown size (<0), it
      /// doesn't have static shape. If all dimensions have known size (>= 0), it
      /// has static shape.
      bool hasStaticShape() const;

      /// If this has a static shape and the shape is equal to `shape` return true.
      bool hasStaticShape(ArrayRef<int64_t> shape) const;

      /// If this is a ranked type, return the number of dimensions with dynamic
      /// size. Otherwise, abort.
      int64_t getNumDynamicDims() const;

      /// If this is ranked type, return the size of the specified dimension.
      /// Otherwise, abort.
      int64_t getDimSize(unsigned idx) const;

      /// Returns true if this dimension has a dynamic size (for ranked types);
      /// aborts for unranked types.
      bool isDynamicDim(unsigned idx) const;

      /// Returns the position of the dynamic dimension relative to just the dynamic
      /// dimensions, given its `index` within the shape.
      unsigned getDynamicDimIndex(unsigned index) const;

      /// Get the total amount of bits occupied by a value of this type.  This does
      /// not take into account any memory layout or widening constraints, e.g. a
      /// vector<3xi57> is reported to occupy 3x57=171 bit, even though in practice
      /// it will likely be stored as in a 4xi64 vector register.  Fail an assertion
      /// if the size cannot be computed statically, i.e. if the type has a dynamic
      /// shape or if its elemental type does not have a known bit width.
      int64_t getSizeInBits() const;

      /// Methods for support type inquiry through isa, cast, and dyn_cast.
      static bool classof(Type type);

      /// Whether the given dimension size indicates a dynamic dimension.
      static constexpr bool isDynamic(int64_t dSize)
      {
        return false;
        // return dSize == kDynamicSize;
      }
      static constexpr bool isDynamicStrideOrOffset(int64_t dStrideOrOffset)
      {
        return false;
        // return dStrideOrOffset == kDynamicStrideOrOffset;
      }
    };

    /// Base MemRef
    class BaseMemRefType : public ShapedType
    {
    public:
      using ShapedType::ShapedType;

      /// Return true if the specified element type is ok in a memref.
      static bool isValidElementType(Type type);

      /// Methods for support type inquiry through isa, cast, and dyn_cast.
      static bool classof(Type type);

      /// Returns the memory space in which data referred to by this memref resides.
      // Attribute getMemorySpace() const;
    };

  } // end tor
} // end mlir

//===----------------------------------------------------------------------===//
// Tablegen Type Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "TOR/TORTypes.h.inc"

#endif // TOR_TYPES_H