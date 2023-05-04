#ifndef TOR_DIALECT_TD
#define TOR_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"

def TORDialect : Dialect {
  let name = "tor";
  let summary = "Types and operations for tor dialect";
  let description = [{
    This dialect defines the 'tor' dialect, which ...
  }];
  let cppNamespace = "::mlir::tor";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    // Register all RTL types.
    void registerTypes();
    void registerAttributes();
  }];
}

class TOROp<string mnemonic, list<Trait> traits = []> :
    Op<TORDialect, mnemonic, traits>;



#endif // TOR_DIALECT_TD
