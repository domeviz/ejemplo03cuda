cmake_minimum_required(VERSION 3.26)
project(ejemplo03 CUDA)

set(CMAKE_CUDA_STANDARD 17)

enable_language(CXX CUDA)

find_package(fmt)
find_package(SFML)

add_executable(ejemplo03
        kernel.cu
        glad.cpp
        main.cpp)

target_include_directories(ejemplo03
        PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(ejemplo03
        fmt::fmt sfml::sfml
        ${CUDART_LIBRARY})

set_target_properties(ejemplo03 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)



