
def NotOp : HECOp<"not",[Pure]> {
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
  let hasFolder = 1;
}

def AddIOp : HECOp<"addi", [Pure]> {
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
def SubIOp : HECOp<"subi", [Pure]> {
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

def MulIConstOp : HECOp<"muli_const", [Pure]> {
  let summary = "HEC muli_const";
  let description = [{
    The "hec.muli_const" operation represents the mulplication of two variables.
    One of them must be a constant.
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

def CmpIOp : HECOp<"cmpi", [Pure]> {
  let summary = "HEC cmpi";
  let description = [{
    The "hec.cmpi" operation represents a integer cmp
  }];
  let arguments = (ins
    AnyType : $lhs,
    AnyType : $rhs,
    StrAttr : $pred,
    Optional<I1> : $guard);
  
  let results = (outs
    I1 : $res
  );
  
  let hasFolder = 1;

  let assemblyFormat= [{
    $pred $lhs $rhs (` ` `if` $guard^)? attr-dict `:` 
    `(` type($lhs) `,` type($rhs) `)` `->` type($res)
  }];
}

def AndOp : HECOp<"and", [Pure, SameOperandsAndResultType]> {
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

def OrOp : HECOp<"or", [Pure, SameOperandsAndResultType]> {
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

def XOrOp : HECOp<"xor", [Pure, SameOperandsAndResultType]> {
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

def ShiftLeftOp : HECOp<"shift_left", [Pure]> {
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
    $lhs $rhs (` ` `if` $guard^)? attr-dict `:` 
    `(` type($lhs) `,` type($rhs) `)` `->` type($res)
  }];
}

def SignedShiftRightOp : HECOp<"shrsi", [Pure]> {
  let summary = "HEC shrsi";
  let description = [{
    The "hec.shrsi" operation represents a logical and
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
    $lhs $rhs (` ` `if` $guard^)? attr-dict `:` 
    `(` type($lhs) `,` type($rhs) `)` `->` type($res)
  }];
}

def UnSignedShiftRightOp : HECOp<"shrui", [Pure]> {
  let summary = "HEC shrui";
  let description = [{
    The "hec.shrui" operation represents a logical and
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
    $lhs $rhs (` ` `if` $guard^)? attr-dict `:` 
    `(` type($lhs) `,` type($rhs) `)` `->` type($res)
  }];
}

def TruncateIOp : HECOp<"trunci", [Pure]> {
  let summary = "HEC trunci";
  let description = [{
    The "hec.trunci" operation represents a logical and
  }];
  let arguments = (ins
    AnyType : $src,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $src (` ` `if` $guard^)? attr-dict `:` type($src) `to` type($res)
  }];
}

def SignExtendIOp: HECOp<"extsi", [Pure]> {
  let summary = "HEC extsi";
  let description = [{
    The "hec.extsi"
  }];
  let arguments = (ins
    AnyType : $src,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $src (` ` `if` $guard^)? attr-dict `:` type($src) `to` type($res)
  }];
}

def ZeroExtendIOp: HECOp<"extui", [Pure]> {
  let summary = "HEC extui";
  let description = [{
    The "hec.extui"
  }];
  let arguments = (ins
    AnyType : $src,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $src (` ` `if` $guard^)? attr-dict `:` type($src) `to` type($res)
  }];
}

def NegFOp: HECOp<"negf", [Pure]> {
  let summary = "HEC negf";
  let description = [{ 
    The "hec.negf"
  }];
  let arguments = (ins
    AnyType : $src,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $src (` ` `if` $guard^)? attr-dict `:` type($src) `to` type($res)
  }];
}

def AbsFOp: HECOp<"absf", [Pure]> {
  let summary = "HEC absf";
  let description = [{ 
    The "hec.absf"
  }];
  let arguments = (ins
    AnyType : $src,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
    $src (` ` `if` $guard^)? attr-dict `:` type($src) `to` type($res)
  }];
}

def AbsIOp: HECOp<"absi", [Pure]> {
  let summary = "HEC absi";
  let description = [{ 
    The "hec.absi"
  }];
  let arguments = (ins
    AnyType : $src,
    Optional<I1> : $guard
  );
  let results = (outs
    AnyType : $res
  );

  let assemblyFormat = [{
     $src (` ` `if` $guard^)? attr-dict `:` type($src) `to` type($res)
  }];
}
 
def SelectOp: HECOp<"select", [Pure, AllTypesMatch<["lhs", "rhs", "res"]>]> {
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
