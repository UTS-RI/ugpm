find_package(Boost COMPONENTS system filesystem iostreams REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_library(libgnuplot-iostream INTERFACE)
target_sources(libgnuplot-iostream INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/gnuplot-iostream.h)
target_link_libraries(libgnuplot-iostream INTERFACE ${Boost_LIBRARIES})
