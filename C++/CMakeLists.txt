cmake_minimum_required(VERSION 3.24)
project(diffrax LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures to build")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(diffrax src/main.cpp src/optimizer.cu)
target_include_directories(diffrax PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(diffrax PRIVATE CUDA::cudart CUDA::cufft ${OpenCV_LIBS})
target_compile_options(diffrax PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--extended-lambda>
)
set_target_properties(diffrax PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS diffrax RUNTIME DESTINATION bin)
