cmake_minimum_required(VERSION 3.13.4)

project(mlir-tor LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

message(STATUS "Using nlohmann_json in: thirdparty/json")

set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(thirdparty/json)
include_directories(thirdparty/json/include)
include_directories(thirdparty/lp_solve_5.5)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

set(LP_SOLVE_PATH "${CMAKE_SOURCE_DIR}/thirdparty/lp_solve_5.5/lpsolve55")

if (WIN32)
    find_library(LP_SOLVE_LIB NAMES liblpsolve55.lib PATHS "${LP_SOLVE_PATH}/bin/win64")
elseif (UNIX)
    if (NOT EXISTS "${LP_SOLVE_PATH}/bin/ux64/liblpsolve55.a")
        execute_process(
                COMMAND sh "${LP_SOLVE_PATH}/ccc"
                WORKING_DIRECTORY ${LP_SOLVE_PATH}
        )
    endif()
    find_library(LP_SOLVE_LIB NAMES liblpsolve55.a PATHS "${LP_SOLVE_PATH}/bin/ux64")
endif ()

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(hector-opt)
#add_subdirectory(hector-translate)
