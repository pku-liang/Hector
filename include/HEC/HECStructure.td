#ifndef HEC_STRUCTURE_TD
#define HEC_STRUCTURE_TD

class HECContainer<string mnemonic, list<OpTrait> traits = []> :
  HECOp<mnemonic, !listconcat(traits, [
    NoRegionArguments,
    NoTerminator,
    SingleBlock
  ])> {
  let regions = (region SizedRegion<1>: $body);
  let assemblyFormat = "attr-dict $body";
  
  let extraClassDeclaration = [{
    // return the body of a HEC container.
    Block *getBody() {
      return &getOperation()->getRegion(0).front();
    }
  }];
}

def DesignOp : HECOp<"design", [
    NoRegionArguments,
    NoTerminator,
    SingleBlock,
    IsolatedFromAbove,
    SymbolTable
  ]> {
  let summary = "HEC Design";
  let description = [{
    The "hec.design" operation represents a top-level HEC design,
    containing a list of HEC component. This must include a "main" component.
  }];
  let arguments = (ins FlatSymbolRefAttr : $symbol);
  let regions = (region SizedRegion<1>: $body);

  let extraClassDeclaration = [{
    // return the body of a HEC design.
    Block *getBody() {
      return &getOperation()->getRegion(0).front();
    }
    
    // Returns the main component, representing the entry point of the design.
    ComponentOp getMainComponent() {
      return lookupSymbol<ComponentOp>("main");
    }
  }];

  let assemblyFormat = "$symbol $body attr-dict";
  let verifier = "return ::verify$cppClass(*this);";
}

def ComponentOp : HECOp<"component", [
    HasParent<"DesignOp">,
    Symbol,
    FunctionLike,
    /*IsolatedFromAbove,*/
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "HEC Component";
  let description = [{
    The "hec.component" operation represents a wrapped/naked component, which is
    presented in either STG manner or Handshake manner.
    Besides, input and output port definitions defines the interface given.
    ```mlir
    hec.component @simple_for {type = "wrapped", style = "STG"} {
      ... // instances
      hec.stateset {}
    }

    hec.component @irregular_for {type = "wrapped", style = "handshake"} {
      ... // instances
      hec.graph {}
    }
    ```
    TODO: Check ports according to interface:
    * `go`, `done` for naked
    * `ready`, `valid` for wrapped
  }];
  let arguments = (ins
    ArrayAttr : $portNames,
    I64Attr : $numInPorts,
    StrAttr : $interfc,
    StrAttr : $style
  );

  let results = (outs);
  let regions = (region SizedRegion<1> : $body);

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<ComponentPortInfo>": $ports,
                   "StringAttr":$interfc, "StringAttr":$style)>
  ];

  let extraClassDeclaration = [{
    using FunctionLike::front;
    private:
      friend class OpTrait::FunctionLike<ComponentOp>;
      unsigned getNumFuncArguments() { return getType().getNumInputs(); }
      unsigned getNumFuncResults() { return getType().getNumResults(); }
    public:
      Block *getBody() {
        return &getOperation()->getRegion(0).front();
      }
      StateSetOp getStateSetOp();
      GraphOp getGraphOp();
  }];
  let verifier = "return ::verify$cppClass(*this);";
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
}

def InstanceOp : HECOp<"instance", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    HasParent<"ComponentOp">
  ]> {
  let summary = "HEC Instance";
  let description = [{
    The "hec.instance" operation represents a instance of a HEC component.

    ```mlir
      %name.in, %name.out = hec.instance "name" @MyComponent : i64, i16
    ```
  }];

  let extraClassDeclaration = [{
    /// Lookup the component for the symbol. This returns null on invalid name.
    ComponentOp getReferencedComponent();
  }];

  let arguments = (ins
    StrAttr:$instanceName,
    FlatSymbolRefAttr:$componentName
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $instanceName $componentName attr-dict (`:` type($results)^)?
  }];
  let verifier = "return ::verify$cppClass(*this);";
}

def PrimitiveOp : HECOp<"primitive", [
  DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ParentOneOf<["ComponentOp", "DesignOp"]>
  ]> {

  let summary = "HEC Primitive Instance";
  let description = [{
    The "hec.primitive" operation represents a instance of a HEC primitive
    component.
    ```mlir
      %reg.in, %reg.out = hec.primitive "reg" < "register" : i32, i32
    ```
  }];

  let extraClassDeclaration = [{
    // Todo: Lookup the primitive component by name.
    SmallVector<ComponentPortInfo> getPrimitivePortInfo();
  }];

  let arguments = (ins
    StrAttr:$instanceName,
    StrAttr:$primitiveName
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $instanceName ` ` `is` $primitiveName attr-dict (`:` type($results)^)? 
  }];
  let verifier = "return ::verify$cppClass(*this);";
}


def AssignOp : HECOp<"assign", [
    /*TypesMatchWith<"dest and src types should be equivalent",
                   "dest", "src", [{ $_self }]>,*/
    ParentOneOf<["StateSetOp", "StateOp", "GraphOp", "StageOp"]>
  ]> {
  
  let summary = "HEC Assign";
  let description = [{
    The "hec.assign" operation represents a non-blocking assignment, which may
    optionally be guarded.
  }];

  let arguments = (ins
    AnyType : $dest,
    AnyType : $src,
    Optional<I1> : $guard
  );
  
  let assemblyFormat = [{
    $dest `=`  $src (`if` $guard^)?  attr-dict `:` type($src) `->` type($dest)
  }];
}

def WireOp : HECOp<"wire",[]> {
  
  let summary = "HEC Wire";
  let description = [{
    The `hec.wire` represents a def of a new wire variable.
  }];

  let arguments = (ins
    StrAttr : $name
  );
  let results = (outs
    AnyType : $out
  );

  let assemblyFormat = [{
    $name attr-dict `:` type($out)
  }];
}

def EnableOp : HECOp<"enable", []> {

  let summary = "HEC Enable";
  let description = [{
      The "hec.enable" represents a signal passing for a port receiving 1.
  }];
  let arguments = (ins
    AnyType : $port,
    Optional<I1> : $cond
  );

  let assemblyFormat = [{
    $port (` ` `if` $cond^)? attr-dict `:` type($port)
  }];
}

def GoOp : HECOp<"go", []> {

  let summary = "HEC Go";
  let description = [{
    The "hec.go" represents the beginning of a instance's execution.
  }];

  let arguments = (ins
    StrAttr : $name, 
    Optional<I1> : $cond
  );
  
  let assemblyFormat = [{
    $name (` ` `if` $cond^)? attr-dict
  }];
}
#endif // HEC_STRUCTURE_TD