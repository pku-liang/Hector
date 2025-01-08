#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/Debug.h"

#include "TOR/PassDetail.h"
#include "TOR/Passes.h"
#include "TOR/TORDialect.h"
#include "TOR/Utils.h"

#include <iostream>
#include <map>
#include <set>

#define DEBUG_TYPE "new-array-partition"

namespace {
using namespace mlir;
using namespace mlir::func;

void warningNonStandardAffineAccess(std::string type, Attribute varNameAttr) {
  llvm::errs() << "warning: " << type << " variable " << varNameAttr
               << " with applying array_partition pragma is failed, "
                  "because non standard affine access.\n";
}

std::string stringAddNumber(std::string str, int number) {
  return str = str + llvm::Twine(number).str();
}

std::string getLineStringWithRank(Value val, int rank) {
  std::string lineString = "array_partition";
  if (auto blockArg = val.dyn_cast<mlir::BlockArgument>()) {
    lineString += stringAddNumber("_arg_", blockArg.getArgNumber());
  }
  return lineString + stringAddNumber("_dim_", rank);
}

std::string getVariableTypeByValue(Value val) {
  if (auto blockArg = val.dyn_cast<mlir::BlockArgument>()) {
    auto funcOp = cast<FuncOp>(blockArg.getOwner()->getParentOp());
    return "@" + funcOp.getSymName().str() + " function with argument";
  }
  return "alloca";
}

Attribute getVarNameTypeByValue(Value val) {
  auto op = getDefiningOpByValue(val);
  if (auto blockArg = val.dyn_cast<mlir::BlockArgument>()) {
    return op->getAttr(stringAddNumber("var_name_", blockArg.getArgNumber()));
  }
  return op->getAttr("var_name");
}

void warningNonStandardAffineAccessByVal(Value val) {
  warningNonStandardAffineAccess(getVariableTypeByValue(val),
                                 getVarNameTypeByValue(val));
}

void warningBankCannotBeDivided(StringRef type, Attribute varNameAttr,
                                int dim) {
  llvm::errs() << "warning: " << type << " variable " << varNameAttr
               << " with applying array_partition pragma on dim " << dim
               << " is failed, because bank cannot be divided.\n";
}

void warningBankCannotBeDividedByVal(Value val, int dim) {
  warningBankCannotBeDivided(getVariableTypeByValue(val),
                             getVarNameTypeByValue(val), dim);
}

int getAttrInterger(Attribute attr) {
  return attr.cast<IntegerAttr>().getValue().getSExtValue();
}

bool hasSymbolic(AffineExpr expr) {
  switch (expr.getKind()) {
  case AffineExprKind::Constant:
    return false;
  case AffineExprKind::DimId:
    return false;
  case AffineExprKind::SymbolId:
    return true;

  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    auto binExpr = expr.cast<AffineBinaryOpExpr>();
    return hasSymbolic(binExpr.getLHS()) || hasSymbolic(binExpr.getRHS());
  }
    llvm_unreachable("Unknown AffineExpr");
  }
}

int getMemBank(AffineMap map, int rank, MLIRContext *ctx, int factor,
               bool cyclic) {
  auto expr = map.getResult(rank);
  if (!expr || hasSymbolic(expr)) {
    return -1;
  }
  if (cyclic) {
    expr = expr % factor;
  } else {
    expr = expr.floorDiv(factor);
  }
  auto compose_map = AffineMap::get(map.getNumDims(), 0, expr, ctx);
  if (compose_map.isConstant()) {
    return compose_map.getConstantResults()[0];
  }
  return -1;
}

int isFullyPartition(ArrayAttr attr) {
  return attr.size() == 1 && getAttrInterger(attr[0]) == -1;
}

void getFactorMapAndCyclicMap(Operation *op, MemRefType memref,
                              DenseMap<int, int> &factorMap,
                              DenseMap<int, bool> &cyclicMap,
                              bool fullyPartition) {
  auto partitionDimArray =
      op->getAttr("partition_dim_array").dyn_cast<ArrayAttr>();
  auto partitionFactorArray =
      op->getAttr("partition_factor_array").dyn_cast<ArrayAttr>();
  auto partitionCyclicArray =
      op->getAttr("partition_cyclic_array").dyn_cast<ArrayAttr>();
  if (fullyPartition) {
    int factor = getAttrInterger(partitionFactorArray[0]);
    for (int i = 0, e = memref.getShape().size(); i < e; i++) {
      if (factor == -1) {
        factorMap[i] = memref.getShape()[i];
      } else {
        factorMap[i] = factor;
      }
      cyclicMap[i] = getAttrInterger(partitionCyclicArray[0]);
    }
    return;
  }
  for (int i = 0, e = partitionDimArray.size(); i < e; i++) {
    int partitionDim = getAttrInterger(partitionDimArray[i]);
    factorMap[partitionDim] = getAttrInterger(partitionFactorArray[i]);
    cyclicMap[partitionDim] = getAttrInterger(partitionCyclicArray[i]);
  }
}

bool checkValueUsers(Value val, int64_t rankNum) {
  for (auto *op : val.getUsers()) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      if (load.getAffineMap().getNumResults() != rankNum) {
        return false;
      }
    } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
      if (store.getAffineMap().getNumResults() != rankNum) {
        return false;
      }
    } else if (auto call = dyn_cast<func::CallOp>(op)) {
      // ignore recursion
      // auto getFuncOpByVal = [](Value val) {
      //   if (auto blockArg = val.dyn_cast<BlockArgument>()) {
      //     return cast<FuncOp>(blockArg.getOwner()->getParentOp());
      //   } else {
      //     return cast<FuncOp>(val.getDefiningOp()->getParentOp());
      //   }
      // };
      // if (getFuncOpByVal(val).getSymName() == call.getCallee()) {
      //   return false;
      // }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Unknown memory operation: " << op << "\n");
      return false;
    }
  }
  return true;
}

bool rankCanBePartition(Value arg, size_t rank, unsigned bank_factor,
                        bool cyclic, MLIRContext *ctx) {
  for (auto *op : arg.getUsers()) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      if (getMemBank(load.getAffineMap(), rank, ctx, bank_factor, cyclic) ==
          -1) {
        LLVM_DEBUG(llvm::dbgs() << "Can not Partition " << load << "\n");
        return false;
      }
    } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
      if (getMemBank(store.getAffineMap(), rank, ctx, bank_factor, cyclic) ==
          -1) {
        LLVM_DEBUG(llvm::dbgs() << "Can not Partition " << store << "\n");
        return false;
      }
    }
  }
  return true;
}

void getPartitionVector(SmallVector<bool> &partition, Value arg,
                        MemRefType memref, DenseMap<int, int> factorMap,
                        DenseMap<int, bool> cyclicMap, StringRef type,
                        Attribute varNameAttr, PatternRewriter &rewriter,
                        bool fullyPartition) {
  for (size_t rank = 0, e = memref.getRank(); rank < e; ++rank) {
    bool flag = false;
    if (factorMap.count(rank)) {
      unsigned bank_factor = cyclicMap[rank]
                                 ? factorMap[rank]
                                 : memref.getShape()[rank] / factorMap[rank];
      flag = rankCanBePartition(arg, rank, bank_factor, cyclicMap[rank],
                                rewriter.getContext());
      std::string lineString = "array_partition";
      if (auto blockArg = arg.dyn_cast<mlir::BlockArgument>()) {
        lineString += "_arg_" + llvm::Twine(blockArg.getArgNumber()).str();
      }
      lineString += "_dim_" + llvm::Twine(rank).str();
      if (!flag) {
        if (fullyPartition) {
          partition.clear();
          return;
        }
      } else {
        // setPragmaStructureAttrStatusByValue(arg, lineString);
      }
    }
    partition.push_back(flag);
  }
  if (fullyPartition) {
    // setPragmaStructureAttrStatusByValue(arg, "array_partition");
  }
}

bool needPartition(SmallVector<bool> partition) {
  for (auto p : partition) {
    if (p) {
      return true;
    }
  }
  return false;
}

int getNewBank(int bank, int index, int factor, bool cyclic, int rankShape) {
  unsigned bank_factor = cyclic ? factor : rankShape / factor;
  int addBank = cyclic ? index % bank_factor : index / bank_factor;
  if (!cyclic && addBank >= factor) {
    return bank * factor + index % factor;
  }
  return bank * factor + addBank;
}

void getInitialValue(MemRefType memref, int rank, int flattenedIndex, int bank,
                     int targetBank, DenseElementsAttr elementsAttr,
                     SmallVector<bool> partition, DenseMap<int, int> factorMap,
                     DenseMap<int, bool> cyclicMap,
                     SmallVector<char> &outDataVec) {
  if (rank == memref.getRank()) {
    if (bank == targetBank) {
      int elementByteSize =
          elementsAttr.getRawData().size() / elementsAttr.getNumElements();
      for (int i = 0; i < elementByteSize; i++) {
        outDataVec.push_back(
            elementsAttr.getRawData()[flattenedIndex * elementByteSize + i]);
      }
    }
    return;
  }
  for (int i = 0; i < memref.getShape()[rank]; i++) {
    int newBank = bank;
    if (partition[rank]) {
      newBank = getNewBank(bank, i, factorMap[rank], cyclicMap[rank],
                           memref.getShape()[rank]);
    }
    getInitialValue(
        memref, rank + 1, flattenedIndex * memref.getShape()[rank] + i, newBank,
        targetBank, elementsAttr, partition, factorMap, cyclicMap, outDataVec);
  }
}

void createNewArray(Operation *op, SmallVector<Value> &newArray,
                    SmallVector<bool> partition, MemRefType memref,
                    DenseMap<int, int> factorMap, PatternRewriter &rewriter,
                    int rank, SmallVector<int64_t> newShape,
                    DenseMap<int, bool> cyclicMap) {
  if (rank == memref.getRank()) {
    auto newMemref = MemRefType::get(newShape, memref.getElementType());
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      funcOp.insertArgument(funcOp.getNumArguments(), newMemref, {},
                            funcOp.getLoc());
      newArray.push_back(funcOp.getArgument(funcOp.getNumArguments() - 1));
    } else if (auto allocOp = dyn_cast<memref::AllocaOp>(op)) {
      auto newAllocOp =
          rewriter.create<memref::AllocaOp>(allocOp->getLoc(), newMemref);
      newArray.push_back(newAllocOp->getResult(0));
    } else if (auto globalOp = dyn_cast<memref::GlobalOp>(op)) {
      Attribute initValue = rewriter.getUnitAttr();
      if (globalOp.getInitialValue().has_value()) {
        if (auto elementsAttr = llvm::dyn_cast<DenseElementsAttr>(
                globalOp.getInitialValue().value())) {
          SmallVector<char> outDataVec;
          getInitialValue(memref, 0, 0, 0, newArray.size(), elementsAttr,
                          partition, factorMap, cyclicMap, outDataVec);
          auto tensorType = RankedTensorType::get(
              newShape, elementsAttr.getType().getElementType());
          initValue =
              DenseElementsAttr::getFromRawBuffer(tensorType, outDataVec);
        }
      }
      std::string newMemrefName =
          (globalOp.getSymName() + "_" + std::to_string(newArray.size())).str();
      auto newGlobalOp = rewriter.create<memref::GlobalOp>(
          globalOp->getLoc(), rewriter.getStringAttr(newMemrefName),
          /*sym_visibility*/ mlir::StringAttr(), mlir::TypeAttr::get(newMemref),
          initValue, mlir::UnitAttr(),
          /*alignment*/ nullptr);
      auto getGlobalOp = rewriter.create<memref::GetGlobalOp>(
          newGlobalOp->getLoc(), newMemref, newGlobalOp.getSymName());
      newArray.push_back(getGlobalOp->getResult(0));
    } else {
      llvm_unreachable("Unknown op");
    }
    return;
  }
  if (partition[rank]) {
    int smallRank = memref.getShape()[rank] / factorMap[rank];
    int bigRankNum = memref.getShape()[rank] % factorMap[rank];
    for (int i = 0; i < bigRankNum; i++) {
      SmallVector<int64_t> iterNewShape(newShape);
      iterNewShape.push_back(smallRank + 1);
      createNewArray(op, newArray, partition, memref, factorMap, rewriter,
                     rank + 1, iterNewShape, cyclicMap);
    }
    if (smallRank) {
      for (int i = bigRankNum; i < factorMap[rank]; i++) {
        SmallVector<int64_t> iterNewShape(newShape);
        iterNewShape.push_back(smallRank);
        createNewArray(op, newArray, partition, memref, factorMap, rewriter,
                       rank + 1, iterNewShape, cyclicMap);
      }
    }
  } else {
    SmallVector<int64_t> iterNewShape(newShape);
    iterNewShape.push_back(memref.getShape()[rank]);
    createNewArray(op, newArray, partition, memref, factorMap, rewriter,
                   rank + 1, iterNewShape, cyclicMap);
  }
}

struct Partition {
  Operation *op;
  unsigned bank;
};

int getDimBank(AffineMap map, int rank, PatternRewriter &rewriter, int factor,
               bool cyclic, int rankShape) {
  unsigned bank_factor = cyclic ? factor : rankShape / factor;
  auto *ctx = rewriter.getContext();
  int addBank = getMemBank(map, rank, ctx, bank_factor, cyclic);
  if (!cyclic && addBank >= factor) {
    return getMemBank(map, rank, ctx, factor, true);
  }
  return addBank;
}

AffineExpr getDimExpr(AffineMap map, int rank, PatternRewriter &rewriter,
                      int factor, bool cyclic, int rankShape) {
  unsigned bank_factor = cyclic ? factor : rankShape / factor;
  auto *ctx = rewriter.getContext();
  int addBank = getMemBank(map, rank, ctx, bank_factor, cyclic);
  auto dimExpr = getAffineDimExpr(rank, ctx);
  auto expr = cyclic ? dimExpr.floorDiv(bank_factor) : dimExpr % bank_factor;
  if (!cyclic && addBank >= factor) {
    int offset = getMemBank(map, rank, ctx, bank_factor, true);
    expr = expr + bank_factor - offset;
  }
  return expr;
}

void getExprs(SmallVector<AffineExpr> &exprs, AffineExpr expr, int rank,
              int rankNum, PatternRewriter &rewriter) {
  for (int i = 0; i < rankNum; ++i) {
    if (i != rank) {
      exprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
    } else {
      exprs.push_back(expr);
    }
  }
}

template <typename T>
void changeAccessOpAttr(T loadOrStore, SmallVector<AffineExpr> exprs,
                        PatternRewriter &rewriter) {
  auto map = loadOrStore.getAffineMap();
  loadOrStore->setAttr(
      loadOrStore.getMapAttrStrName(),
      AffineMapAttr::get(
          AffineMap::get(map.getNumResults(), 0, exprs, rewriter.getContext())
              .compose(map)));
}

template <typename T>
void changeAccessOpAttrFunc(T loadOrStore, int rank, PatternRewriter &rewriter,
                            int factor, bool cyclic, int rankShape,
                            unsigned rankNum) {
  SmallVector<AffineExpr> exprs;
  auto expr = getDimExpr(loadOrStore.getAffineMap(), rank, rewriter, factor,
                         cyclic, rankShape);
  getExprs(exprs, expr, rank, rankNum, rewriter);
  changeAccessOpAttr(loadOrStore, exprs, rewriter);
}

template <typename T>
void calBankAndChangeOpAttr(T loadOrStore, unsigned &bank, int rank,
                            PatternRewriter &rewriter, int factor, bool cyclic,
                            int rankShape, unsigned rankNum) {
  auto map = loadOrStore.getAffineMap();
  bank = bank * factor +
         getDimBank(map, rank, rewriter, factor, cyclic, rankShape);
  changeAccessOpAttrFunc(loadOrStore, rank, rewriter, factor, cyclic, rankShape,
                         rankNum);
}

void changeMemrefAndOperands(Value arg, MemRefType memref,
                             DenseMap<int, int> factorMap,
                             DenseMap<int, bool> cyclicMap,
                             PatternRewriter &rewriter,
                             SmallVector<bool> partition,
                             SmallVector<Value> newArray) {
  SmallVector<Partition> new_part;
  for (auto &use : llvm::make_early_inc_range(arg.getUses())) {
    auto op = use.getOwner();
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      unsigned bank = 0;
      for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
        if (partition[rank]) {
          calBankAndChangeOpAttr(load, bank, rank, rewriter, factorMap[rank],
                                 cyclicMap[rank], memref.getShape()[rank],
                                 memref.getRank());
        }
      }
      new_part.push_back(Partition{load, bank});
    } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
      unsigned bank = 0;
      for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
        if (partition[rank]) {
          calBankAndChangeOpAttr(store, bank, rank, rewriter, factorMap[rank],
                                 cyclicMap[rank], memref.getShape()[rank],
                                 memref.getRank());
        }
      }
      new_part.push_back(Partition{store, bank});
    } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
      auto operands = op->getOperands();
      SmallVector<Value, 4> newOperands;
      for (unsigned int i = 0, e = operands.size(); i < e; i++) {
        if (i != use.getOperandNumber()) {
          newOperands.push_back(operands[i]);
        }
      }
      for (auto val : newArray) {
        newOperands.push_back(val);
      }
      rewriter.setInsertionPoint(callOp);
      rewriter.replaceOpWithNewOp<func::CallOp>(
          callOp, callOp.getCallee(), callOp.getResultTypes(), newOperands);
    }
  }
  for (auto part : new_part) {
    part.op->setOperand(isa<AffineStoreOp>(part.op), newArray[part.bank]);
  }
}

void partitionFunc(Operation *op, SmallVector<bool> partition,
                   MemRefType memref, DenseMap<int, int> factorMap,
                   DenseMap<int, bool> cyclicMap, PatternRewriter &rewriter,
                   Value arg) {
  SmallVector<Value> newArray;
  SmallVector<int64_t> newShape;
  createNewArray(op, newArray, partition, memref, factorMap, rewriter, 0,
                 newShape, cyclicMap);
  changeMemrefAndOperands(arg, memref, factorMap, cyclicMap, rewriter,
                          partition, newArray);
}

struct AllocaOpPattern : OpRewritePattern<memref::AllocaOp> {
  AllocaOpPattern(MLIRContext *ctx) : OpRewritePattern<memref::AllocaOp>(ctx) {}

  LogicalResult matchAndRewrite(memref::AllocaOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("array-partition") || !op->hasAttr("partition_dim_array"))
      return failure();
    op->setAttr("array-partition",
                IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
    auto arg = op->getResult(0);
    auto memref = cast<MemRefType>(arg.getType());
    if (!checkValueUsers(arg, memref.getRank())) {
      // warningNonStandardAffineAccess("alloca", op->getAttr("var_name"));
      return failure();
    }
    DenseMap<int, int> factorMap;
    DenseMap<int, bool> cyclicMap;
    bool fullyPartition = isFullyPartition(
        op->getAttr("partition_dim_array").dyn_cast<ArrayAttr>());
    getFactorMapAndCyclicMap(op, memref, factorMap, cyclicMap, fullyPartition);
    SmallVector<bool> partition;
    getPartitionVector(partition, op, memref, factorMap, cyclicMap, "alloca",
                       op->getAttr("var_name"), rewriter, fullyPartition);
    if (!needPartition(partition)) {
      return success();
    }
    LLVM_DEBUG(llvm::dbgs() << "handle operation: " << op << "\n");
    partitionFunc(op, partition, memref, factorMap, cyclicMap, rewriter, arg);
    return success();
  }
};

void getGlobalPartitionVector(
    SmallVector<bool> &partition, Operation *op,
    SmallVector<memref::GetGlobalOp> &getGlobalOpArray, MemRefType memref,
    DenseMap<int, int> factorMap, DenseMap<int, bool> cyclicMap, StringRef type,
    Attribute varNameAttr, PatternRewriter &rewriter, bool fullyPartition) {
  for (size_t rank = 0, e = memref.getRank(); rank < e; ++rank) {
    bool flag = false;
    if (factorMap.count(rank)) {
      unsigned bank_factor = cyclicMap[rank]
                                 ? factorMap[rank]
                                 : memref.getShape()[rank] / factorMap[rank];
      std::string lineString = "array_partition";
      lineString += "_dim_" + llvm::Twine(rank).str();
      for (auto getGlobalOp : getGlobalOpArray) {
        flag = rankCanBePartition(getGlobalOp, rank, bank_factor,
                                  cyclicMap[rank], rewriter.getContext());
        if (!flag) {
          if (fullyPartition) {
            partition.clear();
            // setPragmaStructureAttrStatusByOp(op, "array_partition", false);
            warningBankCannotBeDivided(type, varNameAttr, 0);
            return;
          }
          warningBankCannotBeDivided(type, varNameAttr, rank + 1);
          // setPragmaStructureAttrStatusByOp(op, lineString, false);
          break;
        }
      }
      if (flag) {
        // setPragmaStructureAttrStatusByOp(op, lineString);
      }
    }
    partition.push_back(flag);
  }
  if (fullyPartition) {
    // setPragmaStructureAttrStatusByOp(op, "array_partition");
  }
}

void getNewGlobalArray(Operation *op, SmallVector<Value> newArray,
                       SmallVector<Value> &newPartitionArray,
                       PatternRewriter &rewriter) {
  for (auto value : newArray) {
    auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(value.getDefiningOp());
    rewriter.setInsertionPoint(op);
    auto newGetGlobalOp = rewriter.create<memref::GetGlobalOp>(
        op->getLoc(), getGlobalOp.getType(), getGlobalOp.getName());
    newPartitionArray.push_back(newGetGlobalOp);
  }
}

void globalPartitionFunc(Operation *op, SmallVector<bool> partition,
                         MemRefType memref, DenseMap<int, int> factorMap,
                         DenseMap<int, bool> cyclicMap,
                         PatternRewriter &rewriter,
                         SmallVector<memref::GetGlobalOp> &getGlobalOpArray) {
  SmallVector<Value> newArray;
  SmallVector<int64_t> newShape;
  createNewArray(op, newArray, partition, memref, factorMap, rewriter, 0,
                 newShape, cyclicMap);
  for (auto getGlobalOp : getGlobalOpArray) {
    SmallVector<Value> newGlobalArray;
    getNewGlobalArray(getGlobalOp, newArray, newGlobalArray, rewriter);
    changeMemrefAndOperands(getGlobalOp, memref, factorMap, cyclicMap, rewriter,
                            partition, newGlobalArray);
  }
  for (auto AI : newArray) {
    AI.getDefiningOp()->erase();
  }
}

struct GlobalOpPattern : OpRewritePattern<memref::GlobalOp> {
  DenseMap<StringRef, SmallVector<memref::GetGlobalOp>> &newGetGlobalOpMap;
  SmallVector<Operation *> &accessOpsToErase;

  GlobalOpPattern(
      MLIRContext *ctx,
      DenseMap<StringRef, SmallVector<memref::GetGlobalOp>> &newGetGlobalOpMap,
      SmallVector<Operation *> &accessOpsToErase)
      : OpRewritePattern<memref::GlobalOp>(ctx),
        newGetGlobalOpMap(newGetGlobalOpMap),
        accessOpsToErase(accessOpsToErase) {}

  LogicalResult matchAndRewrite(memref::GlobalOp op,
                                PatternRewriter &rewriter) const override {

    if (op->hasAttr("array-partition") || !op->hasAttr("partition_dim_array"))
      return failure();
    op->setAttr("array-partition",
                IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
    auto memref = cast<MemRefType>(op.getType());
    auto getGlobalOpArray = newGetGlobalOpMap[op.getSymName()];
    for (auto getGlobalOp : getGlobalOpArray) {
      if (!checkValueUsers(getGlobalOp, memref.getRank())) {
        warningNonStandardAffineAccess("global", op->getAttr("var_name"));
        return failure();
      }
    }
    DenseMap<int, int> factorMap;
    DenseMap<int, bool> cyclicMap;
    bool fullyPartition = isFullyPartition(
        op->getAttr("partition_dim_array").dyn_cast<ArrayAttr>());
    getFactorMapAndCyclicMap(op, memref, factorMap, cyclicMap, fullyPartition);
    SmallVector<bool> partition;
    getGlobalPartitionVector(partition, op, getGlobalOpArray, memref, factorMap,
                             cyclicMap, "global", op->getAttr("var_name"),
                             rewriter, fullyPartition);
    if (!needPartition(partition)) {
      return success();
    }
    LLVM_DEBUG(llvm::dbgs() << "handle operation: " << op << "\n");
    globalPartitionFunc(op, partition, memref, factorMap, cyclicMap, rewriter,
                        getGlobalOpArray);
    for (auto getGlobalOp : getGlobalOpArray) {
      accessOpsToErase.push_back(getGlobalOp);
    }
    accessOpsToErase.push_back(op);
    return success();
  }
};

void getArgFactorMapAndCyclicMap(FuncOp op, MemRefType memref, int argIndex,
                                 DenseMap<int, int> &factorMap,
                                 DenseMap<int, bool> &cyclicMap,
                                 bool fullyPartition) {
  auto partitionDimArray =
      op->getAttr(stringAddNumber("partition_dim_array_", argIndex))
          .dyn_cast<ArrayAttr>();
  auto partitionFactorArray =
      op->getAttr(stringAddNumber("partition_factor_array_", argIndex))
          .dyn_cast<ArrayAttr>();
  auto partitionCyclicArray =
      op->getAttr(stringAddNumber("partition_cyclic_array_", argIndex))
          .dyn_cast<ArrayAttr>();
  if (fullyPartition) {
    int factor = getAttrInterger(partitionFactorArray[0]);
    for (int i = 0, e = memref.getShape().size(); i < e; i++) {
      if (factor == -1) {
        factorMap[i] = memref.getShape()[i];
      } else {
        factorMap[i] = factor;
      }
      cyclicMap[i] = getAttrInterger(partitionCyclicArray[0]);
    }
    return;
  }
  for (int i = 0, e = partitionDimArray.size(); i < e; i++) {
    int partitionDim = getAttrInterger(partitionDimArray[i]);
    factorMap[partitionDim] = getAttrInterger(partitionFactorArray[i]);
    cyclicMap[partitionDim] = getAttrInterger(partitionCyclicArray[i]);
  }
}

struct FuncOpPattern : OpRewritePattern<FuncOp> {
  FuncOpPattern(MLIRContext *ctx) : OpRewritePattern<FuncOp>(ctx) {}

  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("array-partition"))
      return failure();
    op->setAttr("array-partition",
                IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
    for (int i = op.getArguments().size() - 1; i >= 0; i--) {
      auto arg = op.getArgument(i);
      if (!arg.getType().isa<MemRefType>() ||
          !op->hasAttr(stringAddNumber("partition_cyclic_array_", i))) {
        continue;
      }
      auto memref = cast<MemRefType>(arg.getType());
      if (!checkValueUsers(arg, memref.getRank())) {
        // warningNonStandardAffineAccess(
        //     "argument", op->getAttr(stringAddNumber("var_name_", i)));
        continue;
      }
      DenseMap<int, int> factorMap;
      DenseMap<int, bool> cyclicMap;
      bool fullyPartition = isFullyPartition(
          op->getAttr(stringAddNumber("partition_dim_array_", i))
              .dyn_cast<ArrayAttr>());
      getArgFactorMapAndCyclicMap(op, memref, i, factorMap, cyclicMap,
                                  fullyPartition);
      SmallVector<bool> partition;
      getPartitionVector(partition, arg, memref, factorMap, cyclicMap,
                         "argument",
                         op->getAttr(stringAddNumber("var_name_", i)), rewriter,
                         fullyPartition);
      if (!needPartition(partition)) {
        continue;
      }
      partitionFunc(op, partition, memref, factorMap, cyclicMap, rewriter, arg);
      op.eraseArgument(i);
      op->removeAttr(stringAddNumber("partition_dim_array_", i));
      op->removeAttr(stringAddNumber("partition_factor_array_", i));
      op->removeAttr(stringAddNumber("partition_cyclic_array_", i));
    }

    return success();
  }
};

bool checkPartitionVector(Value val, DenseMap<int, int> &factorMap,
                          DenseMap<int, bool> &cyclicMap, bool fullyPartition) {
  auto ctx = getDefiningOpByValue(val)->getContext();
  auto memref = cast<MemRefType>(val.getType());
  for (size_t rank = 0, e = memref.getRank(); rank < e; ++rank) {
    if (factorMap.count(rank)) {
      unsigned bank_factor = cyclicMap[rank]
                                 ? factorMap[rank]
                                 : memref.getShape()[rank] / factorMap[rank];
      std::string lineString = "array_partition";
      if (auto blockArg = val.dyn_cast<mlir::BlockArgument>()) {
        lineString += "_arg_" + llvm::Twine(blockArg.getArgNumber()).str();
      }
      lineString += "_dim_" + llvm::Twine(rank).str();
      if (!rankCanBePartition(val, rank, bank_factor, cyclicMap[rank], ctx)) {
        if (fullyPartition) {
          // setPragmaStructureAttrStatusByValue(val, "array_partition", false);
          warningBankCannotBeDividedByVal(val, 0);
          factorMap.clear();
          return false;
        }
        factorMap.erase(rank);
        // setPragmaStructureAttrStatusByValue(val, lineString, false);
        warningBankCannotBeDividedByVal(val, rank + 1);
      }
    }
  }
  return factorMap.size() != 0;
}

std::string getDimStrWithVal(Value val) {
  if (auto blockArg = val.dyn_cast<mlir::BlockArgument>()) {
    return "_" + llvm::Twine(blockArg.getArgNumber()).str();
  }
  return "";
}

void getFullyPartitionFactorMapAndCyclicMap(Value val,
                                            DenseMap<int, int> &factorMap,
                                            DenseMap<int, bool> &cyclicMap,
                                            int factor, int cyclic) {
  factorMap.clear();
  cyclicMap.clear();
  auto memref = cast<MemRefType>(val.getType());
  for (int i = 0, e = memref.getShape().size(); i < e; i++) {
    if (factor == -1) {
      factorMap[i] = memref.getShape()[i];
    } else {
      factorMap[i] = factor;
    }
    cyclicMap[i] = cyclic;
  }
}

void getFactorMapAndCyclicMapWithVal(Value val, DenseMap<int, int> &factorMap,
                                     DenseMap<int, bool> &cyclicMap) {
  auto op = getDefiningOpByValue(val);
  std::string dimStr = getDimStrWithVal(val);
  auto partitionDimArray =
      op->getAttr("partition_dim_array" + dimStr).dyn_cast<ArrayAttr>();
  auto partitionFactorArray =
      op->getAttr("partition_factor_array" + dimStr).dyn_cast<ArrayAttr>();
  auto partitionCyclicArray =
      op->getAttr("partition_cyclic_array" + dimStr).dyn_cast<ArrayAttr>();
  if (isFullyPartition(partitionDimArray)) {
    getFullyPartitionFactorMapAndCyclicMap(
        val, factorMap, cyclicMap, getAttrInterger(partitionFactorArray[0]),
        getAttrInterger(partitionCyclicArray[0]));
    return;
  }
  for (int i = 0, e = partitionDimArray.size(); i < e; i++) {
    int partitionDim = getAttrInterger(partitionDimArray[i]);
    if (factorMap.count(partitionDim) &&
        factorMap[partitionDim] != getAttrInterger(partitionFactorArray[i])) {
      auto lineStr = getLineStringWithRank(val, partitionDim);
    } else {
      factorMap[partitionDim] = getAttrInterger(partitionFactorArray[i]);
    }
    if (cyclicMap.count(partitionDim) &&
        cyclicMap[partitionDim] != getAttrInterger(partitionCyclicArray[i])) {

    } else {
      cyclicMap[partitionDim] = getAttrInterger(partitionCyclicArray[i]);
    }
  }
}

void setInvalidWithVal(Value val, ArrayAttr partitionDimArray) {
  if (isFullyPartition(partitionDimArray)) {
    // setPragmaStructureAttrStatusByValue(val, "array_partition", false);
  } else {
    std::string lineString = "array_partition";
    if (auto blockArg = val.dyn_cast<mlir::BlockArgument>()) {
      lineString += "_arg_" + llvm::Twine(blockArg.getArgNumber()).str();
    }
    for (int i = 0, e = partitionDimArray.size(); i < e; i++) {
      int partitionDim = getAttrInterger(partitionDimArray[i]);
      // setPragmaStructureAttrStatusByValue(
      //     val, lineString + "_dim_" + llvm::Twine(partitionDim).str(), false);
    }
  }
}

void removeAttrOneGroupValueSet(DenseSet<Value> &idValueSet) {
  // mark all pragma is false
  for (auto val : idValueSet) {
    auto typeStr = "partition_dim_array" + getDimStrWithVal(val);
    if (auto attr = getDefiningOpByValue(val)->getAttr(typeStr)) {
      setInvalidWithVal(val, attr.dyn_cast<ArrayAttr>());
    }
  }

  for (auto val : idValueSet) {
    auto op = getDefiningOpByValue(val);
    std::string valDimStr = getDimStrWithVal(val);
    auto removeAttrWithNameAndSuffix = [](mlir::Operation *op, std::string type,
                                          std::string suffix) {
      op->removeAttr(type + suffix);
    };
    removeAttrWithNameAndSuffix(op, "partition_dim_array", valDimStr);
    removeAttrWithNameAndSuffix(op, "partition_factor_array", valDimStr);
    removeAttrWithNameAndSuffix(op, "partition_cyclic_array", valDimStr);
  }
}

void setAttrOneGroupValueSetWithFully(DenseSet<Value> &idValueSet) {
  // mark all pragma is false exclude fully
  for (auto val : idValueSet) {
    auto typeStr = "partition_dim_array" + getDimStrWithVal(val);
    if (auto attr = getDefiningOpByValue(val)->getAttr(typeStr)) {
      if (isFullyPartition(attr.dyn_cast<ArrayAttr>())) {
        setInvalidWithVal(val, attr.dyn_cast<ArrayAttr>());
      }
    }
  }

  Value fullyVal;
  for (auto val : idValueSet) {
    auto typeStr = "partition_dim_array" + getDimStrWithVal(val);
    if (auto attr = getDefiningOpByValue(val)->getAttr(typeStr)) {
      if (isFullyPartition(attr.dyn_cast<ArrayAttr>())) {
        fullyVal = val;
        break;
      }
    }
  }
  auto fullyOp = getDefiningOpByValue(fullyVal);
  std::string fullyDimStr = getDimStrWithVal(fullyVal);
  for (auto val : idValueSet) {
    auto op = getDefiningOpByValue(val);
    std::string valDimStr = getDimStrWithVal(val);
    auto saveAttrWithNameAndSuffix =
        [](mlir::Operation *newOp, mlir::Operation *op, std::string type,
           std::string newSuffix, std::string suffix) {
          if (auto attr = op->getAttr(type + newSuffix)) {
            newOp->setAttr(type + newSuffix, attr);
          }
        };
    saveAttrWithNameAndSuffix(op, fullyOp, "partition_dim_array", valDimStr,
                              fullyDimStr);
    saveAttrWithNameAndSuffix(op, fullyOp, "partition_factor_array", valDimStr,
                              fullyDimStr);
    saveAttrWithNameAndSuffix(op, fullyOp, "partition_cyclic_array", valDimStr,
                              fullyDimStr);
  }
}

void setInvalidWithDenseMap(Value val, ArrayAttr partitionDimArray,
                            DenseMap<int, int> &factorMap) {
  std::string lineString = "array_partition";
  if (auto blockArg = val.dyn_cast<mlir::BlockArgument>()) {
    lineString += "_arg_" + llvm::Twine(blockArg.getArgNumber()).str();
  }
  for (int i = 0, e = partitionDimArray.size(); i < e; i++) {
    int partitionDim = getAttrInterger(partitionDimArray[i]);
    if (!factorMap.count(partitionDim)) {
      // setPragmaStructureAttrStatusByValue(
      //     val, lineString + "_dim_" + llvm::Twine(partitionDim).str(), false);
    }
  }
}

void setAttrOneGroupValueSetWithFactorMap(DenseSet<Value> &idValueSet,
                                          DenseMap<int, int> &factorMap,
                                          DenseMap<int, bool> &cyclicMap) {
  // mark all pragma is false exclude valid dim
  for (auto val : idValueSet) {
    auto typeStr = "partition_dim_array" + getDimStrWithVal(val);
    if (auto attr = getDefiningOpByValue(val)->getAttr(typeStr)) {
      setInvalidWithDenseMap(val, attr.dyn_cast<ArrayAttr>(), factorMap);
    }
  }

  auto ctx = getDefiningOpByValue(*idValueSet.begin())->getContext();
  SmallVector<mlir::Attribute> dimAttrs, factorAttrs, cyclicAttrs;
  for (auto dimAndFactor : factorMap) {
    auto getIntegerAttr = [](int x, MLIRContext *ctx) {
      return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), x);
    };
    dimAttrs.push_back(getIntegerAttr(dimAndFactor.first, ctx));
    factorAttrs.push_back(getIntegerAttr(dimAndFactor.second, ctx));
    cyclicAttrs.push_back(getIntegerAttr(cyclicMap[dimAndFactor.first], ctx));
  }
  for (auto val : idValueSet) {
    auto op = getDefiningOpByValue(val);
    std::string valDimStr = getDimStrWithVal(val);
    auto setAttrWithNameAndSuffix = [](mlir::Operation *op, std::string type,
                                       std::string suffix, MLIRContext *ctx,
                                       SmallVector<mlir::Attribute> attr) {
      op->setAttr(type + suffix, ArrayAttr::get(ctx, attr));
    };
    setAttrWithNameAndSuffix(op, "partition_dim_array", valDimStr, ctx,
                             dimAttrs);
    setAttrWithNameAndSuffix(op, "partition_factor_array", valDimStr, ctx,
                             factorAttrs);
    setAttrWithNameAndSuffix(op, "partition_cyclic_array", valDimStr, ctx,
                             cyclicAttrs);
  }
}

bool hasFullyInOneGroupValueSet(DenseSet<Value> &idValueSet) {
  for (auto val : idValueSet) {
    auto typeStr = "partition_dim_array" + getDimStrWithVal(val);
    if (auto attr = getDefiningOpByValue(val)->getAttr(typeStr)) {
      if (isFullyPartition(attr.dyn_cast<ArrayAttr>())) {
        return true;
      }
    }
  }
  return false;
}

void getFactorMapInOneGroupValueSet(DenseSet<Value> &idValueSet,
                                    DenseMap<int, int> &factorMap,
                                    DenseMap<int, bool> &cyclicMap) {
  for (auto val : idValueSet) {
    auto typeStr = "partition_dim_array" + getDimStrWithVal(val);
    if (auto attr = getDefiningOpByValue(val)->getAttr(typeStr)) {
      getFactorMapAndCyclicMapWithVal(val, factorMap, cyclicMap);
      if (isFullyPartition(attr.dyn_cast<ArrayAttr>())) {
        return;
      }
    }
  }
}

void handleOneGroupValueSet(DenseSet<Value> &idValueSet) {
  auto rankNum = (cast<MemRefType>((*idValueSet.begin()).getType())).getRank();
  bool allValueUserValid = true;
  for (auto val : idValueSet) {
    if (!checkValueUsers(val, rankNum)) {
      allValueUserValid = false;
      warningNonStandardAffineAccessByVal(val);
    }
  }
  if (!allValueUserValid) {
    // clear all pragma info
    removeAttrOneGroupValueSet(idValueSet);
    return;
  }
  bool fullyPartition = hasFullyInOneGroupValueSet(idValueSet);

  DenseMap<int, int> factorMap;
  DenseMap<int, bool> cyclicMap;
  getFactorMapInOneGroupValueSet(idValueSet, factorMap, cyclicMap);

  for (auto val : idValueSet) {
    if (!checkPartitionVector(val, factorMap, cyclicMap, fullyPartition)) {
      break;
    }
  }

  if (!factorMap.size()) {
    removeAttrOneGroupValueSet(idValueSet);
  } else if (fullyPartition) {
    setAttrOneGroupValueSetWithFully(idValueSet);
  } else {
    setAttrOneGroupValueSetWithFactorMap(idValueSet, factorMap, cyclicMap);
  }
}

void getReverseGraph(ModuleOp moduleOp,
                     DenseMap<StringRef, DenseSet<StringRef>> &reverseGraph) {
  moduleOp.walk([&](FuncOp funcOp) {
    funcOp->walk([&](func::CallOp callOp) {
      // a reverse call graph excluding recursion
      if (funcOp.getSymName() != callOp.getCallee()) {
        if (!reverseGraph.count(callOp.getCallee())) {
          reverseGraph[callOp.getCallee()] = DenseSet<StringRef>();
        }
        reverseGraph[callOp.getCallee()].insert(funcOp.getSymName());
      }
    });
  });
}

// traverse callee
void traverseCalleeArg(Value arg, DenseSet<Value> &idValueSet,
                       DenseMap<StringRef, func::FuncOp> &symNameFuncOpMap) {
  for (auto &use : llvm::make_early_inc_range(arg.getUses())) {
    if (auto callOp = dyn_cast<func::CallOp>(use.getOwner())) {
      auto funcArg = symNameFuncOpMap[callOp.getCallee()].getArgument(
          use.getOperandNumber());
      if (!idValueSet.count(funcArg)) {
        idValueSet.insert(funcArg);
        traverseCalleeArg(funcArg, idValueSet, symNameFuncOpMap);
      }
    }
  }
}

// traverse caller
void traverseCallerArg(Value arg, DenseSet<Value> &idValueSet,
                       DenseMap<StringRef, func::FuncOp> &symNameFuncOpMap,
                       DenseMap<StringRef, DenseSet<StringRef>> reverseGraph) {
  if (auto blockArg = arg.dyn_cast<mlir::BlockArgument>()) {
    auto calleeFuncOp = cast<FuncOp>(blockArg.getOwner()->getParentOp());
    // traverse reverse call
    for (auto callerName : reverseGraph[calleeFuncOp.getSymName()]) {
      auto funcOp = symNameFuncOpMap[callerName];
      funcOp->walk([&](func::CallOp callOp) {
        if (calleeFuncOp.getSymName() == callOp.getCallee()) {
          auto callArg = callOp.getOperand(blockArg.getArgNumber());
          if (!idValueSet.count(callArg)) {
            idValueSet.insert(callArg);
            traverseCallerArg(callArg, idValueSet, symNameFuncOpMap,
                              reverseGraph);
            // need traverse callee arg
            traverseCalleeArg(callArg, idValueSet, symNameFuncOpMap);
          }
        }
      });
    }
  }
}

bool isInCurrentSet(Value arg, DenseMap<int, DenseSet<Value>> &idValueMap) {
  for (auto idValueSet : idValueMap) {
    if (idValueSet.second.count(arg)) {
      return true;
    }
  }
  return false;
}

void getIdValueMap(ModuleOp moduleOp,
                   DenseMap<int, DenseSet<Value>> &idValueMap) {
  DenseMap<StringRef, func::FuncOp> symNameFuncOpMap;
  moduleOp.walk(
      [&](FuncOp funcOp) { symNameFuncOpMap[funcOp.getSymName()] = funcOp; });
  // range is [0, groupCount)
  int groupCount = 0;
  moduleOp.walk([&](memref::AllocaOp AI) {
    if (AI->hasAttr("partition_dim_array")) {
      auto arg = AI->getResult(0);
      idValueMap[groupCount] = DenseSet<Value>();
      idValueMap[groupCount].insert(arg);
      traverseCalleeArg(arg, idValueMap[groupCount], symNameFuncOpMap);
      groupCount += 1;
    }
  });
  DenseMap<StringRef, DenseSet<StringRef>> reverseGraph;
  getReverseGraph(moduleOp, reverseGraph);
  moduleOp.walk([&](FuncOp op) {
    for (int i = op.getArguments().size() - 1; i >= 0; i--) {
      auto arg = op.getArgument(i);
      if (arg.getType().isa<MemRefType>() &&
          op->hasAttr(stringAddNumber("partition_cyclic_array_", i)) &&
          !isInCurrentSet(arg, idValueMap)) {
        idValueMap[groupCount] = DenseSet<Value>();
        idValueMap[groupCount].insert(arg);
        traverseCalleeArg(arg, idValueMap[groupCount], symNameFuncOpMap);
        traverseCallerArg(arg, idValueMap[groupCount], symNameFuncOpMap,
                          reverseGraph);
        groupCount += 1;
      }
    }
  });
}

void colorAndCheckCallGraph(ModuleOp moduleOp) {
  DenseMap<int, DenseSet<Value>> idValueMap;
  getIdValueMap(moduleOp, idValueMap);
  for (int i = 0, e = idValueMap.size(); i < e; i++) {
    handleOneGroupValueSet(idValueMap[i]);
  }
}

struct NewArrayPartitionPass : NewArrayPartitionBase<NewArrayPartitionPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    colorAndCheckCallGraph(moduleOp);
    DenseMap<StringRef, SmallVector<memref::GetGlobalOp>> newGetGlobalOpMap;
    moduleOp.walk([&](memref::GetGlobalOp getGlobalOp) {
      if (!newGetGlobalOpMap.count(getGlobalOp.getName())) {
        newGetGlobalOpMap[getGlobalOp.getName()] =
            SmallVector<memref::GetGlobalOp>();
      }
      newGetGlobalOpMap[getGlobalOp.getName()].push_back(getGlobalOp);
    });
    SmallVector<Operation *> accessOpsToErase;
    moduleOp.walk([&](memref::GlobalOp globalOp) {
      RewritePatternSet patterns(&getContext());
      patterns.insert<GlobalOpPattern>(&getContext(), newGetGlobalOpMap,
                                       accessOpsToErase);
      (void)applyOpPatternsAndFold(globalOp.getOperation(), std::move(patterns));
    });
    for (auto op : accessOpsToErase) {
      op->erase();
    }
    moduleOp.walk([&](FuncOp func) {
      RewritePatternSet patterns(&getContext());
      patterns.insert<FuncOpPattern>(&getContext());
      (void)applyOpPatternsAndFold(func.getOperation(), std::move(patterns));
      func->removeAttr("array-partition");
    });
    SmallVector<Operation *> allocaOps;
    moduleOp.walk([&](memref::AllocaOp AI) { allocaOps.push_back(AI); });
    for (auto op : allocaOps) {
      RewritePatternSet patterns(&getContext());
      patterns.insert<AllocaOpPattern>(&getContext());
      (void)applyOpPatternsAndFold(op, std::move(patterns));
    }
  }
};
} // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createNewArrayPartitionPass() {
  return std::make_unique<NewArrayPartitionPass>();
}

} // namespace mlir
