add_mlir_dialect_library(MLIRTOR
    TORDialect.cpp
    TOR.cpp
    TORTypes.cpp
    TORSchedulePass.cpp
    TORPipelinePartitionPass.cpp
    TORSplitPass.cpp
    TORCheckPass.cpp
    SCFToTOR.cpp
    WidthAnalysisPass.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TOR
    ${RROJECT_SOURCE_DIR}/include/Schedule
    DEPENDS
    MLIRTORIncGen
    MLIRTORPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)
