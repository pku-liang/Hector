def AddIOp : TOROp<"addi", [NoSideEffect]> {
  let summary = "addi operation";
  let description = [{
    The "addi" operation represents the addition of two variables.
  }];

  let arguments = (ins AnySignlessInteger : $lhs, AnySignlessInteger : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs AnySignlessInteger : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  let verifier = ?;
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];

}

def SubIOp : TOROp<"subi", [NoSideEffect]> {
  let summary = "subi operation";
  let description = [{
    The "subi" operation represents the substraction of two variables.
  }];

  let arguments = (ins AnySignlessInteger : $lhs, AnySignlessInteger : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs AnySignlessInteger : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  let verifier = ?;
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];

}

def MulIOp : TOROp<"muli", [NoSideEffect]> {
  let summary = "muli operation";
  let description = [{
    The "muli" operation represents the mulplication of two variables.
  }];

  let arguments = (ins AnySignlessInteger : $lhs, AnySignlessInteger : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs AnySignlessInteger : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  let verifier = ?;
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs)>];

}

def AddFOp : TOROp<"addf", [NoSideEffect, SameTypeOperands, 
                           SameOperandsAndResultType]> {
  let summary = "addf operation";
  let description = [{
    The "addf" operation represents the addition of two variables.
  }];

  let arguments = (ins FloatLike : $lhs, FloatLike : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs FloatLike : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` type($result)
  }];
  let verifier = ?;
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs), [{
      return build($_builder, $_state, lhs.getType(), ValueRange{lhs, rhs},
                   ArrayRef<NamedAttribute>{});
    }]>];

}

def SubFOp : TOROp<"subf", [NoSideEffect, SameTypeOperands, 
                           SameOperandsAndResultType]> {
  let summary = "subf operation";
  let description = [{
    The "subf" operation represents the substraction of two variables.
  }];

  let arguments = (ins FloatLike : $lhs, FloatLike : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs FloatLike : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` type($result)
  }];
  let verifier = ?;
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs), [{
      return build($_builder, $_state, lhs.getType(), ValueRange{lhs, rhs},
                   ArrayRef<NamedAttribute>{});
    }]>];

}

def MulFOp : TOROp<"mulf", [NoSideEffect, SameTypeOperands, 
                           SameOperandsAndResultType]> {
  let summary = "mulf operation";
  let description = [{
    The "mulf" operation represents the mulplication of two variables.
  }];

  let arguments = (ins FloatLike : $lhs, FloatLike : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs FloatLike : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` type($result)
  }];
  let verifier = ?;
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs), [{
      return build($_builder, $_state, lhs.getType(), ValueRange{lhs, rhs},
                   ArrayRef<NamedAttribute>{});
    }]>];

}

def DivFOp : TOROp<"divf", [NoSideEffect, SameTypeOperands, 
                           SameOperandsAndResultType]> {
  let summary = "divf operation";
  let description = [{
    The "divf" operation represents the mulplication of two variables.
  }];

  let arguments = (ins FloatLike : $lhs, FloatLike : $rhs, 
                       I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs FloatLike : $result);

  let assemblyFormat = [{$lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` type($result)
  }];
  let verifier = ?;
  let builders = [
    OpBuilder<(ins "Value" : $lhs, "Value" : $rhs), [{
      return build($_builder, $_state, lhs.getType(), ValueRange{lhs, rhs},
                   ArrayRef<NamedAttribute>{});
    }]>];

}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//
// The predicate indicates the type of the comparison to perform:
// (un)orderedness, (in)equality and less/greater than (or equal to) as
// well as predicates that are always true or false.
def CMPF_P_FALSE   : I64EnumAttrCase<"AlwaysFalse", 0, "false">;
def CMPF_P_OEQ     : I64EnumAttrCase<"OEQ", 1, "oeq">;
def CMPF_P_OGT     : I64EnumAttrCase<"OGT", 2, "ogt">;
def CMPF_P_OGE     : I64EnumAttrCase<"OGE", 3, "oge">;
def CMPF_P_OLT     : I64EnumAttrCase<"OLT", 4, "olt">;
def CMPF_P_OLE     : I64EnumAttrCase<"OLE", 5, "ole">;
def CMPF_P_ONE     : I64EnumAttrCase<"ONE", 6, "one">;
def CMPF_P_ORD     : I64EnumAttrCase<"ORD", 7, "ord">;
def CMPF_P_UEQ     : I64EnumAttrCase<"UEQ", 8, "ueq">;
def CMPF_P_UGT     : I64EnumAttrCase<"UGT", 9, "ugt">;
def CMPF_P_UGE     : I64EnumAttrCase<"UGE", 10, "uge">;
def CMPF_P_ULT     : I64EnumAttrCase<"ULT", 11, "ult">;
def CMPF_P_ULE     : I64EnumAttrCase<"ULE", 12, "ule">;
def CMPF_P_UNE     : I64EnumAttrCase<"UNE", 13, "une">;
def CMPF_P_UNO     : I64EnumAttrCase<"UNO", 14, "uno">;
def CMPF_P_TRUE    : I64EnumAttrCase<"AlwaysTrue", 15, "true">;


def CmpFPredicateAttr : I64EnumAttr<
    "CmpFPredicate", "",
    [CMPF_P_FALSE, CMPF_P_OEQ, CMPF_P_OGT, CMPF_P_OGE, CMPF_P_OLT, CMPF_P_OLE,
     CMPF_P_ONE, CMPF_P_ORD, CMPF_P_UEQ, CMPF_P_UGT, CMPF_P_UGE, CMPF_P_ULT,
     CMPF_P_ULE, CMPF_P_UNE, CMPF_P_UNO, CMPF_P_TRUE]> {
  let cppNamespace = "::mlir::tor";
}

def CmpFOp : TOROp<"cmpf", [NoSideEffect, SameTypeOperands]> {
  let summary = "cmpf operation";
  let description = [{
    The "cmpf" operation represents the comparison of two variables.
  }];

  let arguments = (ins 
      CmpFPredicateAttr : $predicate,
      FloatLike : $lhs, FloatLike : $rhs, 
      I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs BoolLike : $result);

  let assemblyFormat = [{
    $predicate
    $lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` type($lhs) `->` type($result)
  }];
  let verifier = ?;

}


//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

def CMPI_P_EQ  : I64EnumAttrCase<"eq", 0>;
def CMPI_P_NE  : I64EnumAttrCase<"ne", 1>;
def CMPI_P_SLT : I64EnumAttrCase<"slt", 2>;
def CMPI_P_SLE : I64EnumAttrCase<"sle", 3>;
def CMPI_P_SGT : I64EnumAttrCase<"sgt", 4>;
def CMPI_P_SGE : I64EnumAttrCase<"sge", 5>;
def CMPI_P_ULT : I64EnumAttrCase<"ult", 6>;
def CMPI_P_ULE : I64EnumAttrCase<"ule", 7>;
def CMPI_P_UGT : I64EnumAttrCase<"ugt", 8>;
def CMPI_P_UGE : I64EnumAttrCase<"uge", 9>;

def CmpIPredicateAttr : I64EnumAttr<
    "CmpIPredicate", "",
    [CMPI_P_EQ, CMPI_P_NE, CMPI_P_SLT, CMPI_P_SLE, CMPI_P_SGT,
     CMPI_P_SGE, CMPI_P_ULT, CMPI_P_ULE, CMPI_P_UGT, CMPI_P_UGE]> {
  let cppNamespace = "::mlir::tor";
}

def CmpIOp : TOROp<"cmpi", [NoSideEffect]> {
  let summary = "cmpi operation";
  let description = [{
    The "cmpf" operation represents the comparison of two variables.
  }];

  let arguments = (ins 
      CmpIPredicateAttr : $predicate,
      SignlessIntegerLike : $lhs, SignlessIntegerLike : $rhs, 
      I32Attr : $starttime, I32Attr : $endtime);
  let results = (outs BoolLike : $result);

  let assemblyFormat = [{
    $predicate
    $lhs $rhs `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
  let verifier = ?;

}

class AllocLikeOp<string mnemonic,
                  Resource resource,
                  list<OpTrait> traits = []> :
    TOROp<mnemonic,
    !listconcat([
     /* AttrSizedOperandSegments */
    ], traits)> {

  let results = (outs Res<TORAnyMemRef, "", [MemAlloc<resource>]>:$memref);

  let extraClassDeclaration = [{
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];

  let assemblyFormat = [{
     attr-dict `:` type($memref)
  }];

}

def AllocOp : AllocLikeOp<"alloc", DefaultResource> {
  let summary = "memory allocation operation";
  let description = [{
    The `alloc` operation allocates a region of memory, as specified by its
    memref type.

    Example:

    ```mlir
    %0 = tor.alloc() : memref<8x64xf32, 1>
    ```
  }];
}

def LoadOp : TOROp<"load",
     [TypesMatchWith<"result type matches element type of 'memref'",
                     "memref", "result",
                     "$_self.cast<MemRefType>().getElementType()">,
                     MemRefsNormalizable]> {
  let summary = "load operation";
  let description = [{
    The `load` op reads an element from a memref specified by an index. The
    output of load is a new value with the same type as the elements of the
    memref. 
  }];

  let arguments = (ins Arg<TORAnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       I32Attr : $starttime,
                       I32Attr : $endtime,
                       Variadic<AnyTypeOf<[AnySignlessInteger, Index]>>:$indices);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "Value":$memref, "IntegerAttr":$starttime, "IntegerAttr":$endtime,
               CArg<"ValueRange", "{}">:$indices), [{
      auto memrefType = memref.getType().cast<MemRefType>();
      $_state.addOperands(memref);
      $_state.addOperands(indices);
      $_state.addAttribute("starttime", starttime);
      $_state.addAttribute("endtime", endtime);

      $_state.types.push_back(memrefType.getElementType());
    }]>];

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return getMemRef().getType().cast<MemRefType>();
    }

    operand_range getIndices() { return {operand_begin() + 1, operand_end()}; }
  }];

  let assemblyFormat = [{$memref `[` $indices `]` `on` ` ` `(` $starttime `to` $endtime `)` 
      attr-dict `:` type($memref)`[`type($indices)`]`}];
}


def StoreOp : TOROp<"store",
     [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                     "memref", "value",
                     "$_self.cast<MemRefType>().getElementType()">,
                     MemRefsNormalizable]> {
  let summary = "store operation";
  let description = [{
    Store a value to a memref location given by indices. The value stored should
    have the same type as the elemental type of the memref. The number of
    arguments provided within brackets need to match the rank of the memref.
  }];

  let arguments = (ins AnyType:$value,
                       Arg<TORAnyMemRef, "the reference to store to",
                           [MemWrite]>:$memref,
                       I32Attr : $starttime,
                       I32Attr : $endtime,
                       Variadic<AnyTypeOf<[AnySignlessInteger, Index]>>:$indices);

  let builders = [
    OpBuilder<(ins "Value":$valueToStore, "Value":$memref,
                   "IntegerAttr":$starttime, "IntegerAttr":$endtime), [{
      $_state.addOperands(valueToStore);
      $_state.addOperands(memref);
      $_state.addAttribute("starttime", starttime);
      $_state.addAttribute("endtime", endtime);

    }]>];

  let extraClassDeclaration = [{
      Value getValueToStore() { return getOperand(0); }

      Value getMemRef() { return getOperand(1); }
      void setMemRef(Value value) { setOperand(1, value); }
      MemRefType getMemRefType() {
        return getMemRef().getType().cast<MemRefType>();
      }

      operand_range getIndices() {
        return {operand_begin() + 2, operand_end()};
      }
  }];

  let assemblyFormat = [{
    $value `to` $memref `[` $indices `]` `on` ` ` `(` $starttime `to` $endtime `)` 
    attr-dict `:` `(` type($value) `,` type($memref)`[`type($indices) `]` `)`
  }];
}