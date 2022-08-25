get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRTOR
        MLIRHEC
        nlohmann_json::nlohmann_json
        TORSCHEDULE
        ${LP_SOLVE_LIB}
        )
add_llvm_executable(hector-opt hector-opt.cpp)

llvm_update_compile_flags(hector-opt)
message("${LP_SOLVE_LIB}")
target_link_libraries(hector-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(hector-opt)
