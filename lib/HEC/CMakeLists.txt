add_mlir_dialect_library(MLIRHEC
        HECDialect.cpp
        HEC.cpp
        HECTypes.cpp
        HECDumpChiselPass.cpp
		HECDynamicSchedulePass.cpp
        HECGenPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/HEC
        ${PROJECT_SOURCE_DIR}/include/TOR

        DEPENDS
        MLIRHECIncGen
        MLIRHECPassIncGen

        LINK_LIBS PUBLIC
        MLIRIR
		MLIRTOR
        )
