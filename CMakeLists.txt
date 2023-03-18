cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0074 NEW)

project(test-hungarian)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

find_package(Eigen3 REQUIRED 3.3)

include_directories($ENV{doctest_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/inc)
file(GLOB SOURCES "*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE Eigen3::Eigen
)
