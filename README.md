# Hermes: A new HLS tool built on the top of Hector

## Installation

1. Install LLVM/MLIR according to https://mlir.llvm.org/getting_started/
   We use the commit cbc378ecb87e3f31dd5aff91f2a621d500640412

```sh
git checkout cbc378ecb87e3f31dd5aff91f2a621d500640412
```

2. Clone the project
   `git clone https://github.com/pku-liang/Hector.git`

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

5. Chisel templates from https://github.com/xuruifan/Hermes.git

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
build/bin/hector-opt -help
```
