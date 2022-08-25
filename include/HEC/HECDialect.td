#ifndef HEC_DIALECT_TD
#define HEC_DIALECT_TD

include "mlir/IR/OpBase.td"

def HECDialect : Dialect {
  let name = "hec";
  let summary = "Types and operations for hec dialect";
  let description = [{
    This dialect defines the 'hec' dialect, which ...
  }];
  let cppNamespace = "::mlir::hec";
  
  let extraClassDeclaration = [{
    /// Register all hec types.
    void registerTypes();
  }];
}

class HECOp<string mnemonic, list<OpTrait> traits = []> :
    Op<HECDialect, mnemonic, traits>;



#endif // HEC_DIALECT_TD