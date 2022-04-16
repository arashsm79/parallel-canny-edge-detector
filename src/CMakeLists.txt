set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
    ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
set(TARGET_EXEC Main)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
    ${OpenMP_CXX_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

add_executable(${TARGET_EXEC}
    main.cpp
    canny.cpp)

target_link_libraries(${TARGET_EXEC}
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX)

target_include_directories(${TARGET_EXEC}
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/include>)
