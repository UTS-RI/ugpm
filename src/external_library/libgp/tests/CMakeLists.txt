INCLUDE(ExternalProject)

# ----- Download and build gtest -----
ExternalProject_Add(googletest
  GIT_REPOSITORY "https://github.com/google/googletest.git"
  GIT_TAG "release-1.7.0"
  INSTALL_COMMAND ""
  UPDATE_COMMAND ""
  PREFIX "gtest"
)

FIND_PACKAGE(Threads REQUIRED)

# ----- Testing -----
INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/gtest/src/googletest/include)
LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/gtest/src/googletest-build)

# ----- Add sources -----
INCLUDE("Sources.cmake")

ADD_EXECUTABLE(gptest ${LIBGP_TESTS})
TARGET_LINK_LIBRARIES(gptest gp ${CMAKE_THREAD_LIBS_INIT} gtest gtest_main)
