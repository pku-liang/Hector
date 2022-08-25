
def NotOp : HECOp<"not",[]> {
  let summary = "HEC Not";
  let description = [{
    The "hec.not" operation represents a NOT logic.
  }];
  let arguments = (ins
    I1 : $src,
    Optional<I1> : $guard
  );
  let results = (outs
    I1 : $res
  );
  let assemblyFormat = [{
    $src (` ` `if` $guard^)? attr-dict `:` type($src)
  }];
}

def AddIOp : HECOp<"addi", []> {
  let summary = "HEC addi";
  let description = [{
    The "hec.addi" operation represents a integer add
  }];
  let arguments = (ins
    AnyType : $lhs,
    AnyType : $rhs,
    Optional<I1> : $guard);
  
  let results = (outs
    AnyType : $res
  );
  
  let assemblyFormat= [{
    $lhs $rhs (` ` `if` $guard^)? attr-dict `:` 
    `(` type($lhs) `,` type($rhs) `)` `->` type($res)
  }];
  
}
def SubIOp : HECOp<"subi", []> {
  let summary = "HEC subi";
  let description = [{
    The "hec.subi" operation represents a integer sub
  }];
  let arguments = (ins
    AnyType : $lhs,
    AnyType : $rhs,
    Optional<I1> : $guard);
  
  let results = (outs
    AnyType : $res
  );
  
  let assemblyFormat= [{
    $lhs $rhs (` ` `if` $guard^)? attr-dict `:` 
    `(` type($lhs) `,` type($rhs) `)` `->` type($res)
  }];
  
}

def CmpIOp : HECOp<"cmpi", []> {
  let summary = "HEC cmpi";
  let description = [{
    The "hec.cmpi" operation represents a integer cmp
  }];
  let arguments = (ins
    AnyType : $lhs,
    AnyType : $rhs,
    StrAttr : $type,
    Optional<I1> : $guard);
  
  let results = (outs
    I1 : $res
  );
  
  let assemblyFormat= [{
    $type $lhs $rhs (` ` `if` $guard^)? attr-dict `:` 
    `(` type($lhs) `,` type($rhs) `)` `->` type($res)
  }];
}

def AndOp : HECOp<"and", [SameOperandsAndResultType]> {
  let summary = "HEC And";
  let description = [{
    The "hec.and" operation represents a logical and
  }];
  let arguments = (ins
    AnyType : $lhs,
    AnyType : $rhs,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $lhs $rhs (` ` `if` $guard^)? attr-dict `:` type($lhs)
  }];
}

def OrOp : HECOp<"or", [SameOperandsAndResultType]> {
  let summary = "HEC Or";
  let description = [{
    The "hec.or" operation represents a logical and
  }];
  let arguments = (ins
    AnyType : $lhs,
    AnyType : $rhs,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $lhs $rhs (` ` `if` $guard^)? attr-dict `:` type($lhs)
  }];
}

def XOrOp : HECOp<"xor", [SameOperandsAndResultType]> {
  let summary = "HEC Xor";
  let description = [{
    The "hec.xor" operation represents a logical and
  }];
  let arguments = (ins
    AnyType : $lhs,
    AnyType : $rhs,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $lhs $rhs (` ` `if` $guard^)? attr-dict `:` type($lhs)
  }];
}

def ShiftLeftOp : HECOp<"shift_left", [SameOperandsAndResultType]> {
  let summary = "HEC Shift_left";
  let description = [{
    The "hec.shift_left" operation represents a logical and
  }];
  let arguments = (ins
    AnyType : $lhs,
    AnyType : $rhs,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $lhs $rhs (` ` `if` $guard^)? attr-dict `:` type($lhs)
  }];
}

def SignedShiftRightOp : HECOp<"shift_right_signed", [SameOperandsAndResultType]> {
  let summary = "HEC Shift_right_signed";
  let description = [{
    The "hec.shift_right_signed" operation represents a logical and
  }];
  let arguments = (ins
    AnyType : $lhs,
    AnyType : $rhs,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $lhs $rhs (` ` `if` $guard^)? attr-dict `:` type($lhs)
  }];
}

def TruncateIOp : HECOp<"trunci", []> {
  let summary = "HEC Shift_right_signed";
  let description = [{
    The "hec.trunci" operation represents a logical and
  }];
  let arguments = (ins
    AnyType : $lhs,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $lhs (` ` `if` $guard^)? attr-dict `:` type($lhs) `to` type($res)
  }];
}

def SignExtendIOp: HECOp<"sexti", []> {
  let summary = "HEC sexti";
  let description = [{
    The "hec.sexti"
  }];
  let arguments = (ins
    AnyType : $lhs,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $lhs (` ` `if` $guard^)? attr-dict `:` type($lhs) `to` type($res)
  }];
}

def NegFOp: HECOp<"negf", []> {
  let summary = "HEC negf";
  let description = [{ 
    The "hec.negf"
  }];
  let arguments = (ins
    AnyType : $lhs,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $lhs (` ` `if` $guard^)? attr-dict `:` type($lhs) `to` type($res)
  }];
}

def SitofpOp : HECOp<"sitofp", []> {
  let summary = "HEC sitofp";
  let description = [{
    The "hec.fptosi"
  }];
  let arguments = (ins
    AnyType : $lhs,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $lhs (` ` `if` $guard^)? attr-dict `:` type($lhs) `to` type($res)
  }];
}

def FptosiOp: HECOp<"fptosi", []> {
  let summary = "HEC fptosi";
  let description = [{
    The "hec.fptosi"
  }];
  let arguments = (ins
    AnyType : $lhs,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $lhs (` ` `if` $guard^)? attr-dict `:` type($lhs) `to` type($res)
  }];
}

def SelectOp: HECOp<"select", [AllTypesMatch<["lhs", "rhs", "res"]>]> {
  let summary = "HEC select";
  let description = [{
    The "hec.select"
  }];
  let arguments = (ins
    I1 : $cond,
    AnyType : $lhs,
    AnyType : $rhs,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $cond $lhs $rhs (` ` `if` $guard^)? attr-dict `:` type($lhs)
  }];
}

def ActivateOp: HECOp<"activate", [AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = "HEC activate";
  let description = [{

  }];
  let arguments = (ins
    AnyType: $lhs,
    AnyType: $rhs,
    AnyType: $cond
  );
  let assemblyFormat = [{
    $lhs `=` $rhs `by` $cond attr-dict `:` type($lhs) `,` type($cond)
  }];
}
