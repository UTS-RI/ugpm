cmake_minimum_required(VERSION 3.2.2)
project(ugpm)


set(CMAKE_BUILD_TYPE Release)

add_subdirectory(external_library)
add_subdirectory(library)
add_subdirectory(app)
