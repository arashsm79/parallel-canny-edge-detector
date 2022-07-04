cmake_minimum_required(VERSION 3.19)

project(cannyopenmp
	VERSION 1.0
    DESCRIPTION "Canny edge detection with openmp"
    LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(src)
