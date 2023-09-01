cmake_minimum_required(VERSION 3.2.2)
project(ugpm)

# BUILD TYPE
set(CMAKE_BUILD_TYPE Release)

# FIND BOOST
find_package(Boost COMPONENTS program_options REQUIRED)

find_package(Ceres)


# Demonstration of the UGPM
add_executable(ugpm_demo example/ugpm_demo.cpp)
set_target_properties(ugpm_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_include_directories( ugpm_demo PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    )

target_link_libraries( ugpm_demo PUBLIC
    ${Boost_LIBRARIES}
    ${CERES_LIBRARIES}
    )




add_executable(ugpm_tests example/ugpm_random_tests.cpp)
set_target_properties(ugpm_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_include_directories( ugpm_tests PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    )

target_link_libraries( ugpm_tests PUBLIC
    ${Boost_LIBRARIES}
    ${CERES_LIBRARIES}
    )



