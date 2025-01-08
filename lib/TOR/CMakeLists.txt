add_mlir_dialect_library(MLIRTOR
    TORDialect.cpp
    TOR.cpp
    TORTypes.cpp
    SCFDump.cpp
    SCFToTOR.cpp
    TORSchedulePass.cpp
    TORSplitPass.cpp
    TORDump.cpp
    ConvertInput.cpp
    TORAttrs.cpp
    ArrayPartition.cpp
    NewArrayPartition.cpp
    HlsUnroll.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TOR
    ${RROJECT_SOURCE_DIR}/include/Schedule
    DEPENDS
    MLIRTORIncGen
    MLIRTOREnumsIncGen
    MLIRTORPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)
