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
# Helper functions
########################################################################################################################
function(reportFound arg)
  if(arg)
    message(CHECK_PASS "found")
  else()
    message(CHECK_FAIL "not found")
  endif()
endfunction()

########################################################################################################################
# Options and variables
########################################################################################################################
option(AFFT_BUILD_EXAMPLES  "Build examples"                  OFF)
option(AFFT_BUILD_TESTS     "Build tests"                     OFF)
option(AFFT_HEADER_ONLY     "Build as header-only library"    ON)
# option(AFFT_STATIC_LIBS     "Link to static libraries"      OFF)
# option(AFFT_GPU_STATIC_LIBS "Link to static GPU libraries"  ${AFFT_STATIC_LIBS})
# option(AFFT_MODULE          "Build as C++20 module"         OFF)

set(AFFT_MAX_DIM_COUNT 4                 CACHE STRING "Maximum number of dimensions supported by the library, default is 4")
set(AFFT_BACKEND_LIST  "POCKETFFT;VKFFT" CACHE STRING "Semicolon separated list of backends to use, default is POCKETFFT and VKFFT")
set(AFFT_GPU_BACKEND   ""                CACHE STRING "GPU framework to use (CUDA, HIP or OPENCL), default is none (no GPU support)")
set(AFFT_MP_BACKEND    ""                CACHE STRING "Multi process framework to use (MPI), default is none (no MP support)")

########################################################################################################################
# Create target
########################################################################################################################
add_library(afft INTERFACE)
add_library(afft::afft ALIAS afft)

target_compile_features(afft INTERFACE cxx_std_17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(afft INTERFACE AFFT_DEBUG)
endif()

target_compile_definitions(afft INTERFACE AFFT_MAX_DIM_COUNT=${AFFT_MAX_DIM_COUNT})
target_include_directories(afft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(afft SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mdspan/include)
target_include_directories(afft SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/span/include)

if(AFFT_HEADER_ONLY)
  target_compile_definitions(afft INTERFACE AFFT_HEADER_ONLY)
endif()

########################################################################################################################
# Set up GPU target if needed
########################################################################################################################
if(NOT ("${AFFT_GPU_BACKEND}" STREQUAL ""))
  if(AFFT_GPU_BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)
    target_include_directories(afft SYSTEM INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(afft INTERFACE AFFT_GPU_CUDA_TOOLKIT_ROOT_DIR="${CUDAToolkit_ROOT_DIR}") #fixme: is empty
    target_link_libraries(afft INTERFACE CUDA::cuda_driver CUDA::cudart CUDA::nvrtc)
  elseif(AFFT_GPU_BACKEND STREQUAL "HIP")
    find_package(hip REQUIRED)
    target_include_directories(afft SYSTEM INTERFACE ${HIP_INCLUDE_DIRS})
    target_compile_definitions(afft INTERFACE AFFT_GPU_HIP_ROOT_DIR="${hip_ROOT_DIR}") #fixme: is empty
    target_link_libraries(afft INTERFACE hip::host)
    
    if((HIP_PLATFORM STREQUAL "amd") OR (HIP_PLATFORM STREQUAL "hcc"))
      target_compile_definitions(afft INTERFACE AFFT_GPU_HIP_PLATFORM_AMD)
    elseif((HIP_PLATFORM STREQUAL "nvidia") OR (HIP_PLATFORM STREQUAL "nvcc"))
      find_package(CUDAToolkit REQUIRED)
      target_include_directories(afft SYSTEM INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
      target_compile_definitions(afft INTERFACE AFFT_GPU_HIP_PLATFORM_NVIDIA)
      target_link_libraries(afft INTERFACE CUDA::cuda_driver CUDA::cudart CUDA::nvrtc)
    else()
      message(FATAL_ERROR "Invalid HIP platform: ${HIP_PLATFORM}")
    endif()
  elseif(AFFT_GPU_BACKEND STREQUAL "OPENCL")
      find_package(OpenCL REQUIRED)
    target_link_libraries(afft INTERFACE OpenCL::OpenCL)
  else()
    message(FATAL_ERROR "Invalid GPU framework: ${AFFT_GPU_BACKEND}")
  endif()

  string(TOUPPER "${AFFT_GPU_BACKEND}" GPU_BACKEND_UPPER)
  target_compile_definitions(afft INTERFACE AFFT_GPU_BACKEND=${GPU_BACKEND_UPPER})
endif()

########################################################################################################################
# Set up MP target if needed
########################################################################################################################
if(NOT ("${AFFT_MP_BACKEND}" STREQUAL ""))
  set(HAS_MP_SUPPORT TRUE)

  if(AFFT_MP_BACKEND STREQUAL "MPI")
    find_package(MPI REQUIRED)
    target_link_libraries(afft INTERFACE MPI::MPI_CXX)
  else()
    message(FATAL_ERROR "Invalid MP framework: ${AFFT_MP_BACKEND}")
  endif()
else()
  set(HAS_MP_SUPPORT FALSE)
endif()

########################################################################################################################
# Set up backend libraries
########################################################################################################################
foreach(BACKEND ${AFFT_BACKEND_LIST})
  if(BACKEND STREQUAL "CLFFT")
    if(AFFT_GPU_BACKEND STREQUAL "OPENCL")
      status(CHECK_START "Finding clFFT library")
      find_library(CLFFT_LIB NAMES "libclFFT" "clFFT" REQUIRED)
      reportFound(CLFFT_LIB)

      target_link_libraries(afft INTERFACE ${CLFFT_LIB})
    endif()
  elseif(BACKEND STREQUAL "CUFFT")
    if(AFFT_GPU_BACKEND STREQUAL "CUDA")
      if(HAS_MP_SUPPORT AND AFFT_MP_BACKEND STREQUAL "MPI")
        find_package(NVHPC REQUIRED)
        target_link_libraries(afft INTERFACE NVHPC::CUFFTMP)
      else()
        target_link_libraries(afft INTERFACE CUDA::cufft)
      endif()
    endif()
  elseif(BACKEND STREQUAL "FFTW3")
    message(CHECK_START "Finding fftw3 library")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")

    message(CHECK_START "Finding fftw3f library")
    find_library(FFTW3F_LIB NAMES "libfftw3f" "fftw3f")
    reportFound(FFTW3F_LIB)

    message(CHECK_START "Finding fftw3f_threads library")
    find_library(FFTW3F_THREADS_LIB NAMES "libfftw3f_threads" "fftw3f_threads")
    reportFound(FFTW3F_THREADS_LIB)

    message(CHECK_START "Finding fftw3 library")
    find_library(FFTW3_LIB NAMES "libfftw3" "fftw3")
    reportFound(FFTW3_LIB)

    message(CHECK_START "Finding fftw3_threads library")
    find_library(FFTW3_THREADS_LIB NAMES "libfftw3_threads" "fftw3_threads")
    reportFound(FFTW3_THREADS_LIB)

    message(CHECK_START "Finding fftw3l library")
    find_library(FFTW3L_LIB NAMES "libfftw3l" "fftw3l")
    reportFound(FFTW3L_LIB)

    message(CHECK_START "Finding fftw3l_threads library")
    find_library(FFTW3L_THREADS_LIB NAMES "libfftw3l_threads" "fftw3l_threads")
    reportFound(FFTW3L_THREADS_LIB)

    message(CHECK_START "Finding fftw3q library")
    find_library(FFTW3Q_LIB NAMES "libfftw3q" "fftw3q")
    reportFound(FFTW3Q_LIB)

    message(CHECK_START "Finding fftw3q_threads library")
    find_library(FFTW3Q_THREADS_LIB NAMES "libfftw3q_threads" "fftw3q_threads")
    reportFound(FFTW3Q_THREADS_LIB)

    if(FFTW3F_LIB AND FFTW3F_THREADS_LIB)
      target_compile_definitions(afft INTERFACE AFFT_FFTW3_HAS_FLOAT)
      target_link_libraries(afft INTERFACE ${FFTW3F_LIB} ${FFTW3F_THREADS_LIB})
    endif()
    if (FFTW3_LIB AND FFTW3_THREADS_LIB)
      target_compile_definitions(afft INTERFACE AFFT_FFTW3_HAS_DOUBLE)
      target_link_libraries(afft INTERFACE ${FFTW3_LIB} ${FFTW3_THREADS_LIB})
    endif()
    if (FFTW3L_LIB AND FFTW3L_THREADS_LIB)
      target_compile_definitions(afft INTERFACE AFFT_FFTW3_HAS_LONG)
      target_link_libraries(afft INTERFACE ${FFTW3L_LIB} ${FFTW3L_THREADS_LIB})
    endif()
    if (FFTW3Q_LIB AND FFTW3Q_THREADS_LIB)
      target_compile_definitions(afft INTERFACE AFFT_FFTW3_HAS_QUAD)
      target_link_libraries(afft INTERFACE ${FFTW3Q_LIB} ${FFTW3Q_THREADS_LIB})
    endif()

    if(AFFT_MP_BACKEND STREQUAL "MPI")
      message(CHECK_START "Finding fftw3f_mpi library")
      find_library(FFTW3F_MPI_LIB NAMES "libfftw3f_mpi" "fftw3f_mpi")
      reportFound(FFTW3F_MPI_LIB)

      message(CHECK_START "Finding fftw3_mpi library")
      find_library(FFTW3_MPI_LIB NAMES "libfftw3_mpi" "fftw3_mpi")
      reportFound(FFTW3_MPI_LIB)

      message(CHECK_START "Finding fftw3l_mpi library")
      find_library(FFTW3L_MPI_LIB NAMES "libfftw3l_mpi" "fftw3l_mpi")
      reportFound(FFTW3L_MPI_LIB)

      if(FFTW3F_LIB AND FFTW3F_THREADS_LIB AND FFTW3F_MPI_LIB)
        target_compile_definitions(afft INTERFACE AFFT_FFTW3_HAS_MPI_FLOAT)
        target_link_libraries(afft INTERFACE ${FFTW3F_MPI_LIB})
      endif()
      if (FFTW3_LIB AND FFTW3_THREADS_LIB AND FFTW3F_MPI_LIB)
        target_compile_definitions(afft INTERFACE AFFT_FFTW3_HAS_MPI_DOUBLE)
        target_link_libraries(afft INTERFACE ${FFTW3_MPI_LIB})
      endif()
      if (FFTW3L_LIB AND FFTW3L_THREADS_LIB AND FFTW3F_MPI_LIB)
        target_compile_definitions(afft INTERFACE AFFT_FFTW3_HAS_MPI_LONG)
        target_link_libraries(afft INTERFACE ${FFTW3L_MPI_LIB})
      endif()
    endif()

    list(POP_BACK CMAKE_MESSAGE_INDENT)
  elseif(BACKEND STREQUAL "MKL")
    find_package(MKL REQUIRED)
    target_link_libraries(afft INTERFACE MKL::MKL)

    # TODO: MPI support
  elseif(BACKEND STREQUAL "POCKETFFT")
    target_include_directories(afft SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pocketfft)
  elseif(BACKEND STREQUAL "HIPFFT")
    if(AFFT_GPU_BACKEND STREQUAL "HIP")
      find_package(hipfft REQUIRED)
      target_link_libraries(afft INTERFACE hip::hipfft)
    endif()
  elseif(BACKEND STREQUAL "ROCFFT")
    if(AFFT_GPU_BACKEND STREQUAL "HIP")
      find_package(rocfft REQUIRED)
      target_link_libraries(afft INTERFACE roc::rocfft)
    endif()
  elseif(BACKEND STREQUAL "VKFFT")
    if((AFFT_GPU_BACKEND STREQUAL "CUDA") OR (AFFT_GPU_BACKEND STREQUAL "HIP") OR (AFFT_GPU_BACKEND STREQUAL "OPENCL"))
      target_include_directories(afft SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/VkFFT/vkFFT)

      if(AFFT_VKFFT_HAS_DOUBLE_DOUBLE)
        message(CHECK_START "Finding quadmath library")
        find_library(QUADMATH_LIB NAMES "quadmath")
        reportFound(QUADMATH_LIB)

        if(QUADMATH_LIB)
          target_compile_definitions(afft INTERFACE AFFT_VKFFT_HAS_QUADMATH)
          target_link_libraries(afft INTERFACE ${QUADMATH_LIB})
        endif()
      endif()
    endif()
  else()
    message(FATAL_ERROR "Invalid backend: ${BACKEND}")
  endif()
endforeach()

# Convert the list of CPU backends to uppercase and pass it to the target
string(TOUPPER "${AFFT_BACKEND_LIST}" BACKEND_LIST_UPPER)
list(TRANSFORM BACKEND_LIST_UPPER PREPEND "AFFT_BACKEND_" OUTPUT_VARIABLE BACKEND_LIST_UPPER_PREP)
string(REPLACE ";" "|" BACKEND_MASK "${BACKEND_LIST_UPPER_PREP}")
target_compile_definitions(afft INTERFACE AFFT_BACKEND_MASK=${BACKEND_MASK})

########################################################################################################################

# Just a temporary solution
iF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wconversion -Wimplicit-fallthrough")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wconversion -Wimplicit-fallthrough")
endif()

# Should be implemented as a function over all files in the source directory
if(AFFT_BUILD_EXAMPLES)
  add_executable(cpu_dft_1D_C2C_simple examples/cpu/dft_1D_C2C_simple.cpp)
  set_target_properties(cpu_dft_1D_C2C_simple PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/examples")
  target_link_libraries(cpu_dft_1D_C2C_simple PRIVATE afft::afft)

  if(NOT (AFFT_GPU_BACKEND STREQUAL ""))
    if(AFFT_GPU_BACKEND STREQUAL "CUDA")
      set(GPU_EXAMPLES_FRAMEWORK_DIR "cuda")
    elseif(AFFT_GPU_BACKEND STREQUAL "HIP")
      set(GPU_EXAMPLES_FRAMEWORK_DIR "hip")
    elseif(AFFT_GPU_BACKEND STREQUAL "OPENCL")
      set(GPU_EXAMPLES_FRAMEWORK_DIR "opencl")
    else()
      message(FATAL_ERROR "Invalid GPU framework: ${AFFT_GPU_BACKEND}")
    endif()

    add_executable(gpu_dft_1D_C2C_simple "examples/gpu/${GPU_EXAMPLES_FRAMEWORK_DIR}/dft_1D_C2C_simple.cpp")
    set_target_properties(gpu_dft_1D_C2C_simple PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/examples")
    target_link_libraries(gpu_dft_1D_C2C_simple PRIVATE afft::afft)
  endif()
endif()

# Should be implemented as a function over all files in the source directory
if(AFFT_BUILD_TESTS)
  
endif()
