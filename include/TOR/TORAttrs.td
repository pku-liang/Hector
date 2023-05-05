include "mlir/IR/AttrTypeBase.td"

class TorAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TORDialect, name, traits> {
    let mnemonic = attrMnemonic;
}

def DependenceAttr : TorAttr<"Dependence", "dep"> {
    let summary = "Dependence attributes";
    let description = [{
        An dependence attributes for annotating the dependence distance between operations,
        usage 
        %1 = op1 {"dep": [(sig1, d1), (sig2, d2)]}
    }];

    let parameters = (ins ArrayRefParameter<"int">: $signatures, ArrayRefParameter<"int">: $distances);

    let builders = [
        AttrBuilder<(ins "ArrayRef<int>": $signatures,
                         "ArrayRef<int>": $distances), [{
            return $_get($_ctxt, signatures, distances);                                        
        }]>
    ];

    // let genVerifyDecl = 1;

    let assemblyFormat = "`<` `(` $signatures `)` `,` `(` $distances `)` `>`";

    let extraClassDeclaration = [{
        class Builder;
    }];

    let genVerifyDecl = 1;
    
    let skipDefaultBuilders = 1;

    let genAccessors = 1;
}