//===----------------------------------------------------------------------===//
//
// Basic data types for the TOR dialect.
//
//===----------------------------------------------------------------------===//

// Base class for other typedefs. Provides dialact-specific defaults.

class TOR_Type<string name, string baseCppClass = "::mlir::tor::Type"> 
    : TypeDef<TORDialect, name, [], baseCppClass> {
  let mnemonic = ?;
}

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//


def TOR_MemRef : TOR_Type<"MemRef", "mlir::tor::BaseMemRefType"> {
  let summary = "Shaped reference to a region of memory in tor";
  let description = [{

    Syntax(copy from Builtin_MemRef):
    ```
    memref-type ::= `memref` `<` dimension-list-ranked type
                    (`,` layout-specification)? (`,` memory-space)? `>`

    stride-list ::= `[` (dimension (`,` dimension)*)? `]`
    strided-layout ::= `offset:` dimension `,` `strides: ` stride-list
    semi-affine-map-composition ::= (semi-affine-map `,` )* semi-affine-map
    layout-specification ::= semi-affine-map-composition | strided-layout
    memory-space ::= attribute-value
    ```


  }];
  
  let mnemonic = "memref";
  let parameters = (ins
    ArrayRefParameter<"int64_t"> : $shape,
    "Type" : $elementType,
    ArrayRefParameter<"StringAttr">: $property,
    "StringAttr" : $rw /*,
    "Attribute" : $memorySpace */
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>": $shape, "Type": $elementType,
      CArg<"ArrayRef<StringAttr>", "{}">: $property,
      CArg<"StringAttr", "{}">: $rw /*,
      CArg<"Attribute", "{}">: $memorySpace */
    ), [{
      // auto nonIdentityMaps = llvm::make_filter_range(affineMaps,
      //   [](AffineMap map) { return !map.isIdentity(); });
      // Drop default memory space value and replace it with empty attribute.
      // Attribute nonDefaultMemorySpace = 
      //     ::mlir::tor::detail::skipDefaultMemorySpace(memorySpace);
      return $_get(elementType.getContext(), shape, elementType, property,
                   /*llvm::to_vector<4>(nonIdentityMaps),*/
                   rw /*, nonDefaultMemorySpace*/);
    }]>
  ];

  let extraClassDeclaration = [{
    class Builder;
  }];

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

def Bool : I<1>;

def IsTORMemRefTypePred : CPred<"$_self.isa<::mlir::tor::MemRefType>()">;
// A container type is a type that has another type embedded within it.
class TORContainerType<Type etype, Pred containerPred, code elementTypeCall,
                    string descr, string cppClassName = "::mlir::tor::Type"> :
    // set dialect
    // check the container predicate
    // substitute the extracted element into the element type checker.
    DialectType<TORDialect, And<[containerPred,
                SubstLeaves<"$_self", !cast<string>(elementTypeCall),
                etype.predicate>]>,
         descr # " of " # etype.summary # " values", cppClassName> {
  // The type of elements in the container.
  Type elementType = etype;

  // Call to retrieve.
  code getElementTypeCall = elementTypeCall;
}

class TORShapedContainerType<list<Type> allowedTypes,
                          Pred containerPred, string descr,
                          string cppClassName = "::mlir::Type"> :
    TORContainerType<AnyTypeOf<allowedTypes>, containerPred,
                  "$_self.cast<::mlir::tor::ShapedType>().getElementType()", descr,
                  cppClassName>;



// Memrefs are blocks of data with fixed type and rank.
class TORMemRefOf<list<Type> allowedTypes> :
    TORShapedContainerType<allowedTypes, IsTORMemRefTypePred, "memref",
                        "::mlir::tor::MemRefType">;

def TORAnyMemRef : TORMemRefOf<[AnyType]>;
