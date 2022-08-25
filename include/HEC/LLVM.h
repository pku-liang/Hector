
#ifndef HEC_LLVM_H
#define HEC_LLVM_H

// MLIR includes a lot of forward declarations of LLVM types, use them.
#include "mlir/Support/LLVM.h"

// Can not forward declare inline functions with default arguments, so we
// include the header directly.
#include "mlir/Support/LogicalResult.h"

// Forward declarations of classes to be imported in to the circt namespace.
namespace mlir
{
  class ArrayAttr;
  class Attribute;
  class Block;
  class BlockAndValueMapping;
  class BlockArgument;
  class BoolAttr;
  class Builder;
  class NamedAttrList;
  class ConversionPattern;
  class ConversionPatternRewriter;
  class ConversionTarget;
  class DenseElementsAttr;
  class Diagnostic;
  class Dialect;
  class DialectAsmParser;
  class DialectAsmPrinter;
  class DictionaryAttr;
  class ElementsAttr;
  class FileLineColLoc;
  class FlatSymbolRefAttr;
  class FloatAttr;
  class FunctionType;
  class FusedLoc;
  class Identifier;
  class IndexType;
  class InFlightDiagnostic;
  class IntegerAttr;
  class IntegerType;
  class Location;
  class MemRefType;
  class MLIRContext;
  class ModuleOp;
  class ModuleTerminatorOp;
  class MutableOperandRange;
  class NamedAttrList;
  class NoneType;
  class OpAsmDialectInterface;
  class OpAsmParser;
  class OpAsmPrinter;
  class OpBuilder;
  class OperandRange;
  class Operation;
  class OpFoldResult;
  class OpOperand;
  class OpResult;
  class OwningModuleRef;
  class ParseResult;
  class Pass;
  class PatternRewriter;
  class Region;
  class RewritePatternSet;
  class ShapedType;
  class SplatElementsAttr;
  class StringAttr;
  class SymbolRefAttr;
  class SymbolTable;
  class TupleType;
  class Type;
  class TypeAttr;
  class TypeConverter;
  class TypeID;
  class TypeRange;
  class TypeStorage;
  class UnknownLoc;
  class Value;
  class ValueRange;
  class VectorType;
  class WalkResult;
  enum class RegionKind;
  struct CallInterfaceCallable;
  struct LogicalResult;
  struct MemRefAccess;
  struct OperationState;

  template <typename SourceOp>
  struct OpConversionPattern;
  template <typename T>
  class OperationPass;
  template <typename SourceOp>
  struct OpRewritePattern;

  using DefaultTypeStorage = TypeStorage;
  using OpAsmSetValueNameFn = function_ref<void(Value, StringRef)>;
  using NamedAttribute = std::pair<Identifier, Attribute>;

  namespace OpTrait
  {
  }

} // namespace mlir
#endif // HEC_LLVM_H
