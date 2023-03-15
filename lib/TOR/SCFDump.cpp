#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"

#include "TOR/TORDialect.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#include <set>
#include <string>

#define DEBUG_TYPE "dump-scf"


namespace {
	using namespace mlir;
	using std::string;
	namespace dump_scf {
		int attr_num;
		string get_attr() {
			return "op_" + std::to_string(attr_num++);
		}

		struct SCFDumpPass : SCFDumpBase<SCFDumpPass> {
			void runOnOperation() override {
				auto designOp = getOperation();
				designOp.walk([&](Operation* op) {
					op->setAttr("dump", StringAttr::get(&getContext(), get_attr().c_str()));
				});
			}

		};
	}

} // namespace

namespace mlir {

	std::unique_ptr<OperationPass<tor::DesignOp>> createSCFDumpPass() {
		return std::make_unique<dump_scf::SCFDumpPass>();
	}

} // namespace mlir
