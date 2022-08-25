#ifndef HEC_GRAPH_TD
#define HEC_GRAPH_TD

def GraphOp : HECOp<"graph", [
    NoRegionArguments,
    NoTerminator,
    SingleBlock,
    RegionKindInterface,
    HasParent<"ComponentOp">
  ]> {
  let summary = "HEC Graph";
  let description = [{
    The "hec.graph" operation represents a handshake graph.

  }];
  let regions = (region SizedRegion<1> : $body);

  let extraClassDeclaration = [{
    /// Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

  }];
  let assemblyFormat = "$body attr-dict";
  let verifier = "return ::verify$cppClass(*this);";
}

#endif // HEC_GRAPH_TD
