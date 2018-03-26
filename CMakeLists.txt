cmake_minimum_required(VERSION 3.0.0)
project(SSDSLAM)

# The version number.
set (SSD_VERSION_MAJOR 1)
set (SSD_VERSION_MINOR 0)

configure_file (
  "${PROJECT_SOURCE_DIR}/SSDConfig.h.in"
  "${PROJECT_BINARY_DIR}/SSDConfig.h"
  )
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/thirdParty/klt
) 

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
if (TARGET Eigen3::Eigen)
  # Use the imported target
endif (TARGET Eigen3::Eigen)
#find_package(Boost COMPONENTS system thread) 
find_package(OpenCV QUIET)
# flags
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O3 -g -std=c++0x -march=native"
#   "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
)

set(klt_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/klt/klt.cpp
  ${PROJECT_SOURCE_DIR}/src/klt/kltTrackingContext.cpp
  ${PROJECT_SOURCE_DIR}/src/klt/kltTracking.cpp
  ${PROJECT_SOURCE_DIR}/src/klt/convolve.cpp
  ${PROJECT_SOURCE_DIR}/src/JointRadiometicCalib/JRC.cpp
  ${PROJECT_SOURCE_DIR}/src/JointRadiometicCalib/MatrixOperation.cpp
 # ${PROJECT_SOURCE_DIR}/thirdParty/klt/storeFeatures.c
 # ${PROJECT_SOURCE_DIR}/thirdParty/klt/trackFeatures.c
 # ${PROJECT_SOURCE_DIR}/thirdParty/klt/writeFeatures.c
 # ${PROJECT_SOURCE_DIR}/thirdParty/klt/error.c
 # ${PROJECT_SOURCE_DIR}/thirdParty/klt/convolve.c
)

include(CTest)
enable_testing()


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

#compile main library
add_library(klt ${klt_SOURCE_FILES})

#add_executable(SSDSLAM main.cpp)

# decide if we have openCV
if (OpenCV_FOUND)
    message("--- found OpenCV, compiling SSD_opencv library.")
    include_directories( ${OpenCV_INCLUDE_DIRS} )
    set(HAS_OPENCV 1)
    add_executable(SSDSLAM ${PROJECT_SOURCE_DIR}/src/main.cpp)
    target_link_libraries(SSDSLAM klt ${OpenCV_LIBS} Eigen3::Eigen)
endif()
