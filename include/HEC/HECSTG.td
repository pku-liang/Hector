#ifndef HEC_STG_TD
#define HEC_STG_TD

def StateSetOp : HECOp<"stateset", [
    NoRegionArguments,
    NoTerminator,
    SingleBlock,
    HasParent<"ComponentOp">,
    SymbolTable,
    RegionKindInterface
  ]> {
  let summary = "HEC State Set";
  let description = [{
    The "hec.stateset" operation represents a set of states with 
    unconditional/conditional transition;

  }];

  let regions = (region SizedRegion<1> : $body);

  let extraClassDeclaration = [{
    /// Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

  }];
  let assemblyFormat = "$body attr-dict";
  let verifier = "return ::verify$cppClass(*this);";
}

def StateOp : HECOp<"state", [
    NoRegionArguments,
    SingleBlockImplicitTerminator<"TransitionOp">,
    HasParent<"StateSetOp">,
    Symbol
  ]> {
  let summary = "HEC State";
  let description = [{
    The "hec.state" operation represents a state in STG, which has a 
    `hec.transition` as its terminator.
    
  }];

  let arguments = (ins I1Attr : $initial);
  let regions = (region SizedRegion<1> : $body);

  let builders = [
    OpBuilder<(ins "StringAttr" : $name, "IntegerAttr" : $initial)>
  ];

  let verifier = "return ::verify$cppClass(*this);";
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
}

def TransitionOp : HECOp<"transition", [
    NoRegionArguments,
    SingleBlock,
    NoTerminator,
    Terminator,
    HasParent<"StateOp">
  ]> {
  
  let summary = "HEC Transition";
  let description = [{
    The "hec.transtion" operation represents the transition of a state. It
    contains several `hec.goto` or `hec.done` ops, in a if-elif-elif-...-else
    style.
  }];

  let regions = (region SizedRegion<1> : $body);
  let assemblyFormat = "$body attr-dict";
  let verifier = "return ::verify$cppClass(*this);";
}
  
def GotoOp : HECOp<"goto", [
    HasParent<"TransitionOp">
  ]> {
  
  let summary = "HEC Goto";
  let description = [{
    The "hec.goto" operation represents a candidate action in the transition
    of a state. It has an optional Cond, which conresponds to if or elif.
  }];

  let arguments = (ins
    FlatSymbolRefAttr : $dest,
    // I1Attr : $hasCond,
    Optional<I1> : $cond
  );

  let assemblyFormat = [{
    $dest (` ` `if` $cond^)? attr-dict
  }];

  let verifier = "return ::verify$cppClass(*this);";
}

def CDoneOp : HECOp<"cdone", [
    HasParent<"TransitionOp">
  ]> {

  let summary = "HEC CDone";
  let description = [{
    The "hec.cdone" operation represents a conditional completion of the HEC's
    execution
  }];

  let arguments = (ins
    I1 : $cond,
    Variadic<AnyType> : $results
  );

  let assemblyFormat = "attr-dict $cond `?` ($results^ `:` type($results))?";
  let verifier = "return ::verify$cppClass(*this);";
}

def DoneOp : HECOp<"done", [
    HasParent<"TransitionOp">
  ]> {
  
  let summary = "HEC Done";
  let description = [{
    The "hec.done" operation represents a completion of the HEC's
    execution
  }];

  let arguments = (ins
    Variadic<AnyType> : $results
  );

  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
  let verifier = "return ::verify$cppClass(*this);";
}


#endif // HEC_STG_TD