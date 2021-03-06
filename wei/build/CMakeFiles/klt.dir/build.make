# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/build

# Include any dependencies generated for this target.
include CMakeFiles/klt.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/klt.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/klt.dir/flags.make

CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o: CMakeFiles/klt.dir/flags.make
CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o: ../src/frontend/OccupancyGrid.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o"
	g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o -c /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/src/frontend/OccupancyGrid.cpp

CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.i"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/src/frontend/OccupancyGrid.cpp > CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.i

CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.s"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/src/frontend/OccupancyGrid.cpp -o CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.s

CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o.requires:

.PHONY : CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o.requires

CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o.provides: CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o.requires
	$(MAKE) -f CMakeFiles/klt.dir/build.make CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o.provides.build
.PHONY : CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o.provides

CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o.provides.build: CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o


CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o: CMakeFiles/klt.dir/flags.make
CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o: ../src/frontend/VisualFrontend.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o"
	g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o -c /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/src/frontend/VisualFrontend.cpp

CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.i"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/src/frontend/VisualFrontend.cpp > CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.i

CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.s"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/src/frontend/VisualFrontend.cpp -o CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.s

CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o.requires:

.PHONY : CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o.requires

CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o.provides: CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o.requires
	$(MAKE) -f CMakeFiles/klt.dir/build.make CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o.provides.build
.PHONY : CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o.provides

CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o.provides.build: CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o


CMakeFiles/klt.dir/src/demo/klt.cpp.o: CMakeFiles/klt.dir/flags.make
CMakeFiles/klt.dir/src/demo/klt.cpp.o: ../src/demo/klt.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/klt.dir/src/demo/klt.cpp.o"
	g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/klt.dir/src/demo/klt.cpp.o -c /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/src/demo/klt.cpp

CMakeFiles/klt.dir/src/demo/klt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/klt.dir/src/demo/klt.cpp.i"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/src/demo/klt.cpp > CMakeFiles/klt.dir/src/demo/klt.cpp.i

CMakeFiles/klt.dir/src/demo/klt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/klt.dir/src/demo/klt.cpp.s"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/src/demo/klt.cpp -o CMakeFiles/klt.dir/src/demo/klt.cpp.s

CMakeFiles/klt.dir/src/demo/klt.cpp.o.requires:

.PHONY : CMakeFiles/klt.dir/src/demo/klt.cpp.o.requires

CMakeFiles/klt.dir/src/demo/klt.cpp.o.provides: CMakeFiles/klt.dir/src/demo/klt.cpp.o.requires
	$(MAKE) -f CMakeFiles/klt.dir/build.make CMakeFiles/klt.dir/src/demo/klt.cpp.o.provides.build
.PHONY : CMakeFiles/klt.dir/src/demo/klt.cpp.o.provides

CMakeFiles/klt.dir/src/demo/klt.cpp.o.provides.build: CMakeFiles/klt.dir/src/demo/klt.cpp.o


# Object files for target klt
klt_OBJECTS = \
"CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o" \
"CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o" \
"CMakeFiles/klt.dir/src/demo/klt.cpp.o"

# External object files for target klt
klt_EXTERNAL_OBJECTS =

klt: CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o
klt: CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o
klt: CMakeFiles/klt.dir/src/demo/klt.cpp.o
klt: CMakeFiles/klt.dir/build.make
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_videostab.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_ts.a
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_superres.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_stitching.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_contrib.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_nonfree.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_ocl.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_gpu.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_photo.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_objdetect.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_legacy.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_video.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_ml.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_calib3d.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_features2d.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_highgui.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_imgproc.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_flann.so.2.4.13
klt: /home/nvidia/Documents/cmpt742/opencv/build/lib/libopencv_core.so.2.4.13
klt: /usr/local/cuda-9.0/lib64/libnppc.so
klt: /usr/local/cuda-9.0/lib64/libnpps.so
klt: /usr/local/cuda-9.0/lib64/libcufft.so
klt: CMakeFiles/klt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable klt"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/klt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/klt.dir/build: klt

.PHONY : CMakeFiles/klt.dir/build

CMakeFiles/klt.dir/requires: CMakeFiles/klt.dir/src/frontend/OccupancyGrid.cpp.o.requires
CMakeFiles/klt.dir/requires: CMakeFiles/klt.dir/src/frontend/VisualFrontend.cpp.o.requires
CMakeFiles/klt.dir/requires: CMakeFiles/klt.dir/src/demo/klt.cpp.o.requires

.PHONY : CMakeFiles/klt.dir/requires

CMakeFiles/klt.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/klt.dir/cmake_clean.cmake
.PHONY : CMakeFiles/klt.dir/clean

CMakeFiles/klt.dir/depend:
	cd /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/build /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/build /home/nvidia/Documents/cmpt742/SimpleVisualOdometry/wei/build/CMakeFiles/klt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/klt.dir/depend

