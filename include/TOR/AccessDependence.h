#ifndef ACCESS_DEPENDENCE_H
#define ACCESS_DEPENDENCE_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"

namespace mlir {

enum OptMode : int {
  OPT_NONE = 0,
  OPT_CONSERVATIVE = 1,
  OPT_AGGRESSIVE = 3,
};

struct MemoryAccessCompare {
  bool hasStore;
  Operation *opInst;
  Value memref;
  SmallVector<Value, 4> indices;
  MemoryAccessCompare() : hasStore(false){};

  Value getMemRef() { return memref; }
};

struct MemRefAccessCompare : public MemoryAccessCompare {
  MemRefAccessCompare(Operation *loadOrStoreOpInst) : MemoryAccessCompare() {
    if (auto loadOp = dyn_cast<memref::LoadOp>(loadOrStoreOpInst)) {
      opInst = loadOrStoreOpInst;
      memref = loadOp.getMemref();
      llvm::append_range(indices, loadOp.getIndices());
    } else {
      assert(isa<memref::StoreOp>(loadOrStoreOpInst) &&
             "Memref load/store op expected");
      auto storeOp = cast<memref::StoreOp>(loadOrStoreOpInst);
      opInst = loadOrStoreOpInst;
      memref = storeOp.getMemref();
      llvm::append_range(indices, storeOp.getIndices());
    }
  };

  bool operator==(const MemRefAccessCompare &rhs) const {
    return std::equal(indices.begin(), indices.end(), rhs.indices.begin());
  }

  static auto getMemoryStoreOp(Operation &loadOrStoreOpInst) {
    return dyn_cast<memref::StoreOp>(loadOrStoreOpInst);
  }

  static auto getMemoryLoadOp(Operation &loadOrStoreOpInst) {
    return dyn_cast<memref::LoadOp>(loadOrStoreOpInst);
  }

  static auto getMemoryLoadResult(Operation *loadOrStoreOpInst) {
    return dyn_cast<memref::LoadOp>(loadOrStoreOpInst)->getResult(0);
  }

  static auto getAnotherMemoryStoreOp(Operation &loadOrStoreOpInst) {
    return dyn_cast<AffineWriteOpInterface>(loadOrStoreOpInst);
  }

  static Value getAnotherMemoryStoreMemRef(Operation &loadOrStoreOpInst) {
    return getAnotherMemoryStoreOp(loadOrStoreOpInst).getMemRef();
  }
};

struct AffineAccessCompare : public MemoryAccessCompare {
  AffineValueMap affineMap;

  AffineAccessCompare(Operation *loadOrStoreOpInst) : MemoryAccessCompare() {
    if (auto loadOp = dyn_cast<AffineReadOpInterface>(loadOrStoreOpInst)) {
      opInst = loadOrStoreOpInst;
      llvm::append_range(indices, loadOp.getMapOperands());
      memref = loadOp.getMemRef();
    } else {
      assert(isa<AffineWriteOpInterface>(loadOrStoreOpInst) &&
             "Affine read/write op expected");
      opInst = loadOrStoreOpInst;
      auto storeOp = cast<AffineWriteOpInterface>(opInst);
      llvm::append_range(indices, storeOp.getMapOperands());
      memref = storeOp.getMemRef();
    }
    getAccessMap(&affineMap);
  };

  void getAccessMap(AffineValueMap *accessMap) const {
    // Get affine map from AffineLoad/Store.
    AffineMap map;
    if (auto loadOp = dyn_cast<AffineReadOpInterface>(opInst))
      map = loadOp.getAffineMap();
    else
      map = cast<AffineWriteOpInterface>(opInst).getAffineMap();

    SmallVector<Value, 8> operands(indices.begin(), indices.end());
    fullyComposeAffineMapAndOperands(&map, &operands);
    map = simplifyAffineMap(map);
    canonicalizeMapAndOperands(&map, &operands);
    accessMap->reset(map, operands);
  }

  bool operator==(const AffineAccessCompare &rhs) const {
    SmallVector<Value, 4> allOperands;
    allOperands.reserve(affineMap.getNumOperands() +
                        rhs.affineMap.getNumOperands());
    auto aDims = affineMap.getOperands().take_front(affineMap.getNumDims());
    auto bDims =
        rhs.affineMap.getOperands().take_front(rhs.affineMap.getNumDims());
    auto aSyms = affineMap.getOperands().take_back(affineMap.getNumSymbols());
    auto bSyms =
        rhs.affineMap.getOperands().take_back(rhs.affineMap.getNumSymbols());
    allOperands.append(aDims.begin(), aDims.end());
    allOperands.append(bDims.begin(), bDims.end());
    allOperands.append(aSyms.begin(), aSyms.end());
    allOperands.append(bSyms.begin(), bSyms.end());
    // shift dims and symbols of rhs.affineMap's map.
    auto bMap = rhs.affineMap.getAffineMap()
                    .shiftDims(affineMap.getNumDims())
                    .shiftSymbols(affineMap.getNumSymbols());

    // construct the difference expressions.
    auto aMap = affineMap.getAffineMap();
    SmallVector<AffineExpr, 4> diffExprs;
    diffExprs.reserve(affineMap.getNumResults());
    for (unsigned i = 0, e = bMap.getNumResults(); i < e; ++i)
      diffExprs.push_back(aMap.getResult(i) - bMap.getResult(i));

    auto diffMap = AffineMap::get(bMap.getNumDims(), bMap.getNumSymbols(),
                                  diffExprs, bMap.getContext());
    // XXX：it can be make expr more equal, but now it can't effect
    fullyComposeAffineMapAndOperands(&diffMap, &allOperands);
    canonicalizeMapAndOperands(&diffMap, &allOperands);
    diffMap = simplifyAffineMap(diffMap);
    return llvm::all_of(diffMap.getResults(),
                        [](AffineExpr e) { return e == 0; });
  }

  static auto getMemoryStoreOp(Operation &loadOrStoreOpInst) {
    return dyn_cast<AffineWriteOpInterface>(loadOrStoreOpInst);
  }

  static auto getMemoryLoadOp(Operation &loadOrStoreOpInst) {
    return dyn_cast<AffineReadOpInterface>(loadOrStoreOpInst);
  }

  static auto getMemoryLoadResult(Operation *loadOrStoreOpInst) {
    return dyn_cast<AffineReadOpInterface>(loadOrStoreOpInst)->getResult(0);
  }

  static auto getAnotherMemoryStoreOp(Operation &loadOrStoreOpInst) {
    return dyn_cast<memref::StoreOp>(loadOrStoreOpInst);
  }

  static Value getAnotherMemoryStoreMemRef(Operation &loadOrStoreOpInst) {
    return getAnotherMemoryStoreOp(loadOrStoreOpInst).getMemref();
  }
};

template <typename T> struct MemeoryAccesses {
  SmallVector<T, 16> accesses;

  void push_back(T memeoryAccess) { accesses.push_back(memeoryAccess); }

  void clear() { accesses.clear(); }

  int find1DAccessindex(T memeoryAccess) {
    int numAccesses = accesses.size();
    for (int i = numAccesses - 1; i >= 0; i--) {
      if (accesses[i] == memeoryAccess) {
        return i;
      }
    }
    return -1;
  }

  Operation *getOpInst(int idx) { return accesses[idx].opInst; }

  bool getStoreStatus(int idx) { return accesses[idx].hasStore; }

  void setMemeoryAccess(int idx, T memeoryAccess) {
    accesses[idx] = memeoryAccess;
  }

  void setStoreStatus(int idx) { accesses[idx].hasStore = true; }
};

template <typename T> struct RemoveMemeoryAccessesAdaptor {
  Block *block;
  SmallVector<Operation *, 8> accessOpsToErase;
  DenseMap<Value, MemeoryAccesses<T>> memrefStoresMap;
  DenseMap<Value, MemeoryAccesses<T>> memrefLoadsMap;
  size_t eraseNum = 0;
  OptMode optMode;

  RemoveMemeoryAccessesAdaptor(Block *block, OptMode optMode)
      : block(block), optMode(optMode) {
    accessOpsToErase.clear();
    memrefStoresMap.clear();
    memrefLoadsMap.clear();
  };

  bool isSingleMemRef(Value AI) {
    if (auto memRefType = AI.getType().dyn_cast<MemRefType>()) {
      for (auto dim : memRefType.getShape()) {
        if (dim != 1) {
          return false;
        }
      }
    } else {
      return false;
    }
    return true;
  }

  void
  insertMemeoryAccessesToMap(DenseMap<Value, MemeoryAccesses<T>> &accessesMap,
                             Value memref) {
    if (!accessesMap.count(memref)) {
      accessesMap[memref] = MemeoryAccesses<T>();
    }
  }

  void removeOpsToErase(SmallVector<Operation *, 8> &accessOpsToErase) {
    eraseNum += accessOpsToErase.size();
    for (auto *op : accessOpsToErase)
      op->erase();
    accessOpsToErase.clear();
  }

  void clearMemrefAccess(Value memref,
                         DenseMap<Value, MemeoryAccesses<T>> &memrefAccessMap) {
    insertMemeoryAccessesToMap(memrefAccessMap, memref);
    auto &memrefAccess = memrefAccessMap[memref];
    memrefAccess.clear();
  }

  void handleDstAccessInRRSWL(T dstAccess) {
    insertMemeoryAccessesToMap(memrefStoresMap, dstAccess.getMemRef());
    auto &memrefStores = memrefStoresMap[dstAccess.getMemRef()];
    int findIndex = memrefStores.find1DAccessindex(dstAccess);
    if (findIndex == -1) {
      memrefStores.push_back(dstAccess);
    } else {
      // add redundant store to accessOpsToErase
      accessOpsToErase.push_back(memrefStores.getOpInst(findIndex));
      // load value should be new store
      memrefStores.setMemeoryAccess(findIndex, dstAccess);
    }
  }

  void handleSrcAccessInRRSWL(T srcAccess) {
    insertMemeoryAccessesToMap(memrefStoresMap, srcAccess.getMemRef());
    auto &memrefStores = memrefStoresMap[srcAccess.getMemRef()];
    int findIndex = memrefStores.find1DAccessindex(srcAccess);
    if (findIndex != -1) {
      srcAccess.opInst->getResult(0).replaceAllUsesWith(
          memrefStores.getOpInst(findIndex)->getOperand(0));
      // add load which aftered store to accessOpsToErase
      accessOpsToErase.push_back(srcAccess.opInst);
    }
  }

  void conservativeHandleDstAccessInRRSWL(T dstAccess) {
    insertMemeoryAccessesToMap(memrefStoresMap, dstAccess.getMemRef());
    auto &memrefStores = memrefStoresMap[dstAccess.getMemRef()];
    int findIndex = memrefStores.find1DAccessindex(dstAccess);
    if (findIndex == -1) {
      clearMemrefAccess(dstAccess.getMemRef(), memrefStoresMap);
      memrefStores.push_back(dstAccess);
    } else {
      // add redundant store to accessOpsToErase
      accessOpsToErase.push_back(memrefStores.getOpInst(findIndex));
      // load value should be new store
      memrefStores.setMemeoryAccess(findIndex, dstAccess);
    }
  }

  void conservativeRemoveRedundantStoreWithinLoad() {
    for (Operation &op : *block) {
      if (auto storeOp = T::getMemoryStoreOp(op)) {
        conservativeHandleDstAccessInRRSWL(T(storeOp));
      } else if (auto loadOp = T::getMemoryLoadOp(op)) {
        handleSrcAccessInRRSWL(T(loadOp));
      } else if (T::getAnotherMemoryStoreOp(op)) {
        clearMemrefAccess(T::getAnotherMemoryStoreMemRef(op), memrefStoresMap);
      } else if (op.getNumRegions() || isa<func::CallOp>(op)) {
        memrefStoresMap.clear();
      }
    }
    // remove redundant store and load which aftered store
    removeOpsToErase(accessOpsToErase);
  }

  void removeRedundantStoreWithinLoad() {
    for (Operation &op : *block) {
      if (auto storeOp = T::getMemoryStoreOp(op)) {
        if (isa<memref::StoreOp>(op) &&
            !isSingleMemRef(storeOp->getOperand(1))) {
          conservativeHandleDstAccessInRRSWL(T(storeOp));
        } else {
          handleDstAccessInRRSWL(T(storeOp));
        }
      } else if (auto loadOp = T::getMemoryLoadOp(op)) {
        handleSrcAccessInRRSWL(T(loadOp));
      } else if (T::getAnotherMemoryStoreOp(op)) {
        clearMemrefAccess(T::getAnotherMemoryStoreMemRef(op), memrefStoresMap);
      } else if (op.getNumRegions() || isa<func::CallOp>(op)) {
        memrefStoresMap.clear();
      }
    }
    // remove redundant store and load which aftered store
    removeOpsToErase(accessOpsToErase);
  }

  void handleDstAccessInRRL(T dstAccess) {
    insertMemeoryAccessesToMap(memrefLoadsMap, dstAccess.getMemRef());
    auto &memrefLoads = memrefLoadsMap[dstAccess.getMemRef()];
    int findIndex = memrefLoads.find1DAccessindex(dstAccess);
    if (findIndex != -1) {
      memrefLoads.setStoreStatus(findIndex);
    }
  }

  void handleSrcAccessInRRL(T srcAccess) {
    insertMemeoryAccessesToMap(memrefLoadsMap, srcAccess.getMemRef());
    auto &memrefLoads = memrefLoadsMap[srcAccess.getMemRef()];
    int findIndex = memrefLoads.find1DAccessindex(srcAccess);
    if (findIndex == -1) {
      memrefLoads.push_back(srcAccess);
    } else if (memrefLoads.getStoreStatus(findIndex)) {
      // behind store flush and old load is invalid value
      memrefLoads.setMemeoryAccess(findIndex, srcAccess);
    } else {
      auto oldLoadResult =
          T::getMemoryLoadResult(memrefLoads.getOpInst(findIndex));
      // use old load result
      srcAccess.opInst->getResult(0).replaceAllUsesWith(oldLoadResult);
      // add repeat load to accessOpsToErase
      accessOpsToErase.push_back(srcAccess.opInst);
    }
  }

  void conservativeHandleDstAccessInRRL(T dstAccess) {
    insertMemeoryAccessesToMap(memrefLoadsMap, dstAccess.getMemRef());
    auto &memrefLoads = memrefLoadsMap[dstAccess.getMemRef()];
    int findIndex = memrefLoads.find1DAccessindex(dstAccess);
    if (findIndex != -1) {
      memrefLoads.setStoreStatus(findIndex);
    } else {
      clearMemrefAccess(dstAccess.getMemRef(), memrefLoadsMap);
    }
  }

  void conservativeRemoveRepeatLoad() {
    for (Operation &op : *block) {
      if (auto storeOp = T::getMemoryStoreOp(op)) {
        conservativeHandleDstAccessInRRL(T(storeOp));
      } else if (auto loadOp = T::getMemoryLoadOp(op)) {
        handleSrcAccessInRRL(T(loadOp));
      } else if (T::getAnotherMemoryStoreOp(op)) {
        clearMemrefAccess(T::getAnotherMemoryStoreMemRef(op), memrefLoadsMap);
      } else if (op.getNumRegions() || isa<func::CallOp>(op)) {
        memrefLoadsMap.clear();
      }
    }
    // remove repeat load
    removeOpsToErase(accessOpsToErase);
  }

  void removeRepeatLoad() {
    for (Operation &op : *block) {
      if (auto storeOp = T::getMemoryStoreOp(op)) {
        if (isa<memref::StoreOp>(op) &&
            !isSingleMemRef(storeOp->getOperand(1))) {
          conservativeHandleDstAccessInRRL(T(storeOp));
        } else {
          handleDstAccessInRRL(T(storeOp));
        }
      } else if (auto loadOp = T::getMemoryLoadOp(op)) {
        handleSrcAccessInRRL(T(loadOp));
      } else if (T::getAnotherMemoryStoreOp(op)) {
        clearMemrefAccess(T::getAnotherMemoryStoreMemRef(op), memrefLoadsMap);
      } else if (op.getNumRegions() || isa<func::CallOp>(op)) {
        memrefLoadsMap.clear();
      }
    }
    // remove repeat load
    removeOpsToErase(accessOpsToErase);
  }

  void removeRedundantAccessByBlock() {
    switch (optMode) {
    case OptMode::OPT_NONE:
      break;
    case OptMode::OPT_CONSERVATIVE: {
      conservativeRemoveRedundantStoreWithinLoad();
      conservativeRemoveRepeatLoad();
      break;
    }
    case OptMode::OPT_AGGRESSIVE: {
      removeRedundantStoreWithinLoad();
      removeRepeatLoad();
      break;
    }
    default:
      break;
    }
  }
};

struct RemoveMemeoryAccesses {
  OptMode optMode;
  Block *block;
  size_t eraseNum;

  RemoveMemeoryAccesses(Block *block, OptMode optMode)
      : optMode(optMode), block(block){};

  void removeAccesses() {
    if (optMode == OptMode::OPT_NONE) {
      return;
    }
    auto memrefAccess =
        RemoveMemeoryAccessesAdaptor<MemRefAccessCompare>(block, optMode);
    memrefAccess.removeRedundantAccessByBlock();
    auto affineAccess =
        RemoveMemeoryAccessesAdaptor<AffineAccessCompare>(block, optMode);
    affineAccess.removeRedundantAccessByBlock();
    eraseNum = memrefAccess.eraseNum + affineAccess.eraseNum;
  }
};

} // namespace mlir
#endif
