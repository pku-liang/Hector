#ifndef TOR_DIALECT_TD
#define TOR_DIALECT_TD

include "mlir/IR/OpBase.td"

def TORDialect : Dialect {
  let name = "tor";
  let summary = "Types and operations for tor dialect";
  let description = [{
    This dialect defines the 'tor' dialect, which ...
  }];
  let cppNamespace = "::mlir::tor";
  
  let extraClassDeclaration = [{
    /// Register all RTL types.
    void registerTypes();
  }];
}

class TOROp<string mnemonic, list<OpTrait> traits = []> :
    Op<TORDialect, mnemonic, traits>;



#endif // TOR_DIALECT_TD