cmake_minimum_required(VERSION 2.8.3)

project(simple_visual_odometry)

set(EIGEN_MALLOC_ALREADY_ALIGNED 0)
set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -o2 -fopenmp")
#set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3 -fopenmp -Wall")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#file(GLOB_RECURSE FIVE_POINT_SOURCE five-point-nister/*.cpp)
#add_library(five-point-nister SHARED ${FIVE_POINT_SOURCE})

# Build the tracking node
file(GLOB_RECURSE VO_SOURCE 
          ./src/backend/*.cpp ./src/frontend/*.cpp ./src/logic/*.cpp)

# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
#set(OpenCV_DIR /usr/share/OpenCV)
set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/../../opencv/build)
find_package(OpenCV 2 REQUIRED)

# Pangolin
#set(Pangolin_DIR ${PROJECT_SOURCE_DIR}/../Pangolin/build)
find_package(Pangolin REQUIRED)

# Sophus
#set(Sophus_DIR ${PROJECT_SOURCE_DIR}/../Sophus/build)
find_package(Sophus REQUIRED)

# G2O
#set(G2O_DIR ${PROJECT_SOURCE_DIR}/../g2o/build)
find_package(G2O REQUIRED)

include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${G2O_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS})
include_directories(include five-point-nister ${EIGEN3_INCLUDE_DIR})
include_directories(./include)

set(THIRD_PARTY_LIBS ${Sophus_LIBRARIES} g2o_core g2o_stuff g2o_types_sba)

add_executable(
	demo 
	./src/frontend/OccupancyGrid.cpp 
	./src/frontend/VisualFrontend.cpp
	./src/demo/demo.cpp 
	)
add_executable(
	klt 
	./src/frontend/OccupancyGrid.cpp
	./src/frontend/VisualFrontend.cpp
	./src/demo/klt.cpp)
add_executable(feature_detector ./src/demo/feature_detector.cpp)
add_executable(klt_five_points ./src/demo/klt_five_points.cpp)
#add_executable(
#    main
#    ./src/frontend/OccupancyGrid.cpp
#    ./src/frontend/VisualFrontend.cpp 
#    ./src/offline/main.cpp)

target_link_libraries(demo ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${THIRD_PARTY_LIBS}
        ${PROJECT_SOURCE_DIR}/src/logic/libfeature_tracking_hidden.so
	${PROJECT_SOURCE_DIR}/src/logic/libfive-point-nister.so)
target_link_libraries(feature_detector ${OpenCV_LIBS})
target_link_libraries(klt ${OpenCV_LIBS})
target_link_libraries(klt_five_points ${OpenCV_LIBS})
#target_link_libraries(main ${OpenCV_LIBS})
