#include "HEC/HECTypes.h"
#include "HEC/HECDialect.h"

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
#include "HEC/HECTypes.cpp.inc"

namespace mlir
{
  namespace hec
  {

  } // end hec
} // end mlir

namespace mlir
{
  namespace hec
  {

    void HECDialect::registerTypes()
    {
      addTypes<
#define GET_TYPEDEF_LIST
#include "HEC/HECTypes.cpp.inc"
          >();
    }

    /// Parses a type registered to this dialect. Parse out the mnemonic then invoke
    /// the tblgen'd type parser dispatcher.
    // Type HECDialect::parseType(DialectAsmParser &parser) const
    // {
    //   llvm::StringRef mnemonic;
    //   if (parser.parseKeyword(&mnemonic))
    //     return Type();
    //   Type type;
    //   auto parseResult = generatedTypeParser(getContext(), parser, mnemonic, type);
    //   if (parseResult.hasValue())
    //     return type;
    //   return Type();
    // }

    // /// Print a type registered to this dialect. Try the tblgen'd type printer
    // /// dispatcher then fail since all RTL types are defined via ODS.
    // void HECDialect::printType(Type type, DialectAsmPrinter &printer) const
    // {
    //   if (succeeded(generatedTypePrinter(type, printer)))
    //     return;
    //   llvm_unreachable("unexpected 'hec' type");
    // }

    namespace detail
    {
    } // end detail

  }
}