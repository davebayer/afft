########################################################################################################################
#
# This file is part of afft library.
#
# Copyright (c) 2024 David Bayer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
########################################################################################################################

cmake_minimum_required(VERSION 3.20)
project(afft VERSION 0.1.0 LANGUAGES CXX)

########################################################################################################################
# Options and variables
########################################################################################################################
option(AFFT_BUILD_EXAMPLES  "Build examples"               OFF)
option(AFFT_BUILD_TESTS     "Build tests"                  OFF)
# option(AFFT_STATIC_LIBS     "Link to static libraries"     OFF)
# option(AFFT_GPU_STATIC_LIBS "Link to static GPU libraries" ${AFFT_STATIC_LIBS})

set(AFFT_CPU_BACKEND_LIST "POCKETFFT" CACHE STRING "Semicolon separated list of CPU backends to use (FFTW3, MKL and POCKETFFT), default is POCKETFFT")
set(AFFT_GPU_FRAMEWORK    ""          CACHE STRING "GPU framework to use (CUDA or HIP), default is none (no GPU support)")
set(AFFT_GPU_BACKEND_LIST "VKFFT"     CACHE STRING "Semicolon separated list of GPU backends to use (CUFFT, HIPFFT, ROCFFT and VKFFT), default is VKFFT")

########################################################################################################################
# Create target
########################################################################################################################
add_library(afft INTERFACE)
add_library(afft::afft ALIAS afft)

target_include_directories(afft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(afft INTERFACE cxx_std_20)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(afft INTERFACE AFFT_DEBUG)
endif()

target_include_directories(afft SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mdspan/include/mdspan)

########################################################################################################################
# Set up CPU target
########################################################################################################################
foreach(CPU_BACKEND ${AFFT_CPU_BACKEND_LIST})
  if(CPU_BACKEND STREQUAL "FFTW3")
    find_library(FFTW3F_LIB NAMES "libfftw3f" "fftw3f" REQUIRED)
    find_library(FFTW3_LIB NAMES "libfftw3" "fftw3" REQUIRED)
    find_library(FFTW3L_LIB NAMES "libfftw3l" "fftw3l")
    find_library(FFTW3Q_LIB NAMES "libfftw3q" "fftw3q")
    find_library(FFTW3F_THREADS_LIB NAMES "libfftw3f_threads" "fftw3f_threads" REQUIRED)
    find_library(FFTW3_THREADS_LIB NAMES "libfftw3_threads" "fftw3_threads" REQUIRED)
    find_library(FFTW3L_THREADS_LIB NAMES "libfftw3l_threads" "fftw3l_threads")
    find_library(FFTW3Q_THREADS_LIB NAMES "libfftw3q_threads" "fftw3q_threads")

    if (FFTW3L_LIB AND FFTW3L_THREADS_LIB)
      target_compile_definitions(afft INTERFACE AFFT_CPU_FFTW3_LONG_FOUND)
    endif()
    if (FFTW3Q_LIB AND FFTW3Q_THREADS_LIB)
      target_compile_definitions(afft INTERFACE AFFT_CPU_FFTW3_QUAD_FOUND)
    endif()

    target_link_libraries(afft INTERFACE ${FFTW3F_LIB} ${FFTW3_LIB} ${FFTW3F_THREADS_LIB} ${FFTW3_THREADS_LIB})
  elseif(CPU_BACKEND STREQUAL "MKL")
    find_package(MKL REQUIRED)
    target_link_libraries(afft INTERFACE MKL::MKL)
  elseif(CPU_BACKEND STREQUAL "POCKETFFT")
    target_include_directories(afft SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pocketfft)
  else()
    message(FATAL_ERROR "Invalid CPU backend: ${CPU_BACKEND}")
  endif()
endforeach()

# Convert the list of CPU backends to uppercase and pass it to the target
string(TOUPPER ${AFFT_CPU_BACKEND_LIST} CPU_BACKEND_LIST_UPPER)
target_compile_definitions(afft INTERFACE AFFT_CPU_BACKEND_LIST=${CPU_BACKEND_LIST_UPPER})

########################################################################################################################
# Set up GPU target if needed
########################################################################################################################
if(NOT (AFFT_GPU_FRAMEWORK STREQUAL ""))
  if(AFFT_GPU_FRAMEWORK STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)

    # Get the CUDA Toolkit root directory from the CUDAToolkit binary directory and pass it to the target
    cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH AFFT_GPU_CUDA_TOOLKIT_ROOT_DIR)
    target_compile_definitions(afft INTERFACE AFFT_GPU_CUDA_TOOLKIT_ROOT_DIR="${AFFT_GPU_CUDA_TOOLKIT_ROOT_DIR}")
    target_link_libraries(afft INTERFACE CUDA::cuda_driver CUDA::cudart CUDA::nvrtc)
  elseif(AFFT_GPU_FRAMEWORK STREQUAL "HIP")
    find_package(hip REQUIRED)
    target_link_libraries(afft INTERFACE hip::host hip::hiprtc)
  else()
    message(FATAL_ERROR "Invalid GPU framework: ${AFFT_GPU_FRAMEWORK}")
  endif()
  
  foreach(GPU_BACKEND ${AFFT_GPU_BACKEND_LIST})
    if(GPU_BACKEND STREQUAL "CUFFT")
      target_link_libraries(afft INTERFACE CUDA::cufft)
    elseif(GPU_BACKEND STREQUAL "HIPFFT")
      target_link_libraries(afft INTERFACE hip::hipfft)
    elseif(GPU_BACKEND STREQUAL "ROCFFT")
      target_link_libraries(afft INTERFACE hip::rocfft)
    elseif(GPU_BACKEND STREQUAL "VKFFT")
      target_include_directories(afft SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/VkFFT/vkFFT)
    else()
      message(FATAL_ERROR "Invalid GPU backend: ${GPU_BACKEND}")
    endif()
  endforeach()

  # Convert the GPU framework and the list of GPU backends to uppercase and pass it to the target
  string(TOUPPER ${AFFT_GPU_FRAMEWORK} GPU_FRAMEWORK_UPPER)
  string(TOUPPER ${AFFT_GPU_BACKEND_LIST} GPU_BACKEND_LIST_UPPER)
  target_compile_definitions(afft INTERFACE AFFT_GPU_FRAMEWORK=${GPU_FRAMEWORK_UPPER})
  target_compile_definitions(afft INTERFACE AFFT_GPU_BACKEND_LIST=${GPU_BACKEND_LIST_UPPER})
endif()

########################################################################################################################

# Just a temporary solution
iF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

# Should be implemented as a function over all files in the source directory
if(AFFT_BUILD_EXAMPLES)
  add_executable(example_dft_1D_C2C_simple examples/dft_1D_C2C_simple.cpp)
  set_target_properties(example_dft_1D_C2C_simple PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
  target_link_libraries(example_dft_1D_C2C_simple PRIVATE afft::afft)
endif()

# Should be implemented as a function over all files in the source directory
if(AFFT_BUILD_TESTS)
  add_executable(test tests/test.cpp)
  target_link_libraries(test PRIVATE afft::afft)
endif()
