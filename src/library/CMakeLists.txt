set(TARGET_NAME celib)

# FIND OPENMP
find_package(OpenMP)

find_package(Ceres)

file(GLOB source src/*/*.cpp)

add_library( ${TARGET_NAME} STATIC ${source} )
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES)
target_include_directories( ${TARGET_NAME} PUBLIC
    include
    ${CERES_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/external_library)

target_link_libraries( ${TARGET_NAME}
    ${CERES_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    gp )

target_compile_options( ${TARGET_NAME} PRIVATE
    ${OpenMP_CXX_FLAGS})

