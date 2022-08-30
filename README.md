# HECTOR: A Multi-level Intermediate Representation for Hardware Synthesis Methodologies

## Introduction
Hardware synthesis requires a complicated process to generate synthesizable register transfer level (RTL) code. High-level synthesis tools can automatically transform a high-level description into hardware design, while hardware generators adopt domain specific languages and synthesis flows for specific applications. The implementation of these tools generally requires substantial engineering efforts due to RTL’s weak expressivity and low level of abstraction. Furthermore, different synthesis tools adopt different levels of intermediate representations (IR) and transformations. A unified IR obviously is a good way to lower the engineering cost and get competitive hardware design rapidly by exploring different synthesis methodologies.

This project proposes Hector, a two-level IR providing a unified intermediate representation for hardware synthesis methodologies. The high-level IR binds computation with a control graph annotated with timing information, while the low-level IR provides a concise way to describe hardware modules and elastic interconnections among them. Implemented based on the multi-level compiler infrastructure (MLIR), Hector’s IRs can be converted to synthesizable RTL designs.

## Installation
1. Install LLVM/MLIR according to https://mlir.llvm.org/getting_started/

2. Clone the project
```git clone https://github.com/pku-liang/Hector.git```

3. Get the submodules
```sh
cd Hector
git submodule update --init --recursive
```
4. Configuration and build
```sh
mkdir build
cd build
cmake -G Ninja .. -DMLIR_DIR=<LLVM_DIR>/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=<LLVM_DIR>/build/bin/llvm-lit
ninja
```

5. Chisel templates from https://github.com/xuruifan/hector_template/

## Run examples
1. Static & Dynamic scheduling in high-level synthesis
```sh
zsh examples/hls_script.sh build/bin/hector-opt examples
```
2. Hybrid scheduling
```sh
cd build
bin/hector-opt ../examples/hybrid-tuned/aeloss_pull.mlir --scf-to-tor --schedule-tor --split-schedule --generate-hec --dynamic-schedule --dump-chisel
```

## Transformation passes
```sh
build\bin\hector-opt -help
```

