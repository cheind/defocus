cmake_minimum_required(VERSION 2.8)

project(defocus)

if (WIN32)
	add_definitions("-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
endif()

set(DEFOCUS_LINK_TARGETS)

set(DEFOCUS_EIGEN_DIR "../eigen" CACHE PATH "Where is the include directory of Eigen located")
if (WIN32)
  set(OpenCV_STATIC OFF)
  set(OpenCV_SHARED ON)
endif()
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
list(APPEND DEFOCUS_LINK_TARGETS ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${DEFOCUS_EIGEN_DIR} "inc")

add_library(defocus
    inc/defocus/features.h
    inc/defocus/camera.h
    inc/defocus/sparse.h
    inc/defocus/io.h
    src/features.cpp
    src/camera.cpp
    src/sparse.cpp
    src/io.cpp
)

add_executable(defocus_tests
    tests/catch.hpp
    tests/test_sfm.cpp
)
target_link_libraries(defocus_tests defocus ${DEFOCUS_LINK_TARGETS})

add_executable(defocus_example
    examples/defocus_example.cpp
)
target_link_libraries(defocus_example defocus ${DEFOCUS_LINK_TARGETS})

add_executable(defocus_statistics
    examples/defocus_statistics.cpp
)
target_link_libraries(defocus_statistics defocus ${DEFOCUS_LINK_TARGETS})