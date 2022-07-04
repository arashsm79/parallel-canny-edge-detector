set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
    ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
set(TARGET_EXEC Main)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS})

add_executable(${TARGET_EXEC}
    main.cu
    canny.cu)

target_link_libraries(${TARGET_EXEC}
    ${OpenCV_LIBS})

target_include_directories(${TARGET_EXEC}
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/include>)

target_compile_options(${TARGET_EXEC} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                      -arch=sm_50;
                     >)
