#ifndef HEC_PIPELINE_TD
#define HEC_PIPELINE_TD

def StageSetOp : HECOp<"stageset", [
    NoRegionArguments,
    NoTerminator,
    SingleBlock,
    HasParent<"ComponentOp">,
    SymbolTable,
    RegionKindInterface
  ]> {
  let summary = "HEC Stages for Loop Pipelining";
  let description = [{
    The "hec.stageset" operation represents a set of cycle-sensitive stages.
  }];

  let regions = (region SizedRegion<1> : $body);

  let extraClassDeclaration = [{
    /// Implement RegionindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph;}
  }];

  let assemblyFormat = "$body attr-dict";
  let verifier = "return ::verify$cppClass(*this);";
}

def StageOp : HECOp<"stage", [
    NoRegionArguments,
    SingleBlock, 
    NoTerminator,
    HasParent<"StageSetOp">,
    Symbol
  ]> {
  let summary = "HEC Stage";
  let description = [{
    The "hec.stage" operation represents a stage in a pipelined loop body.
  }];
  let builders = [
    OpBuilder<(ins "StringAttr" : $name)>
  ];
  let regions = (region SizedRegion<1>: $body);
  
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  let verifier = "return ::verify$cppClass(*this);";
}

def InitOp : HECOp<"init", []> {
  let summary = "HEC Init";
  let description = [{
    The "hec.init" operation represents the initialization of stage-reg in 
    pipelining.
  }];
  let arguments = (ins
    AnyType : $dst,
    AnyType : $src
  );

  let assemblyFormat = "$dst `=` $src attr-dict `:` type($src) `->` type($dst)";
  let verifier = ?;
}

def YieldOp : HECOp<"yield",[]> {
  let summary = "HEC Yield";
  let description = [{
    The "hec.yield" operation represents a value yielding in pipeline stage.
  }];
  let arguments = (ins Variadic<AnyType> : $results);
  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
  let verifier = ?;
}


def DeliverOp : HECOp<"deliver",[]> {
  let summary = "HEC Deliver";
  let description = [{
    The "hec.deliver" operation represents a value yielding in pipeline stage.
  }];
  let arguments = (ins 
    AnyType : $src,
    AnyType : $destReg,
    AnyType : $destPort, 
    Optional<I1>: $guard);

  let assemblyFormat = "$src `to` $destReg `(` $destPort `)` attr-dict  (`if` $guard^)?`:` type($src) `->` type($destReg) `,` type($destPort)";
  let verifier = ?;
}

#endif // HEC_PIPELINE_TD