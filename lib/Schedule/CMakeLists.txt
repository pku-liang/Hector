add_mlir_dialect_library(TORSCHEDULE
    ScheduleAlgo.cpp
    SDCSolver.cpp
    SDCSchedule.cpp
    CDFG.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TOR
    ${RROJECT_SOURCE_DIR}/include/Schedule

    LINK_LIBS PUBLIC
    MLIRIR
)
