##
#  CMake script for the step-26 tutorial program:
##

# Set the name of the project and target:
SET(TARGET "mars_heat")


SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

SET(GCC_COVERAGE_COMPILE_FLAGS "-I/home/basinuser/BasinUser/Dependencies/armadillo-10.6.2/include -L/usr/lib/x86_64-linux-gnu -larmadillo -I/home/basinuser/BasinUser/Dependencies/libconfig-1.7.3/include -L/usr/local/lib -lconfig++ --std=c++17 -lstdc++fs")

SET(GCC_COVERAGE_LINK_FLAGS     "-I/home/basinuser/BasinUser/Dependencies/libconfig-1.7.3/include -L/usr/local/lib -lconfig++ -I/home/basinuser/BasinUser/Dependencies/armadillo-10.6.2/include  -L/usr/lib/x86_64-linux-gnu -Wl,-rpath -L/usr/lib/x86_64-linux-gnu -larmadillo")

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}" )

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
add_definitions(${CMAKE_EXE_LINKER_FLAGS})

FILE(GLOB_RECURSE TARGET_SRC  "support_code/*.cpp")
FILE(GLOB_RECURSE TARGET_INC  "support_code/*.h")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#  FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#  FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#  SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
SET(TARGET_SRC
  ${TARGET}.cc
  )

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

FIND_PACKAGE(deal.II 9.1.0 
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
