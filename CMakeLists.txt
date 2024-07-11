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
project(afft VERSION 0.1.0 LANGUAGES C CXX)

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
option(AFFT_BUILD_EXAMPLES  "Build examples"                                          OFF)
option(AFFT_BUILD_TESTS     "Build tests"                                             OFF)
option(AFFT_USE_NVHPC_CUDA  "Use CUDA version that comes with NVHPC"                  OFF)
option(AFFT_USE_NVHPC_MPI   "Use MPI version that comes with NVHPC"                   OFF)
option(AFFT_ENABLE_CUFFTMP  "Enable multi-process support for cuFFT (requires NVHPC)" OFF)
option(AFFT_USE_NVHPC_CUFFT "Use cuFFT version that comes with NVHPC"                 ${AFFT_ENABLE_CUFFTMP})
option(AFFT_USE_CUDA_OPENCL "Use OpenCL version that comes with CUDA"                 OFF)

# option(AFFT_STATIC_LIBS     "Link to static libraries"              OFF)
# option(AFFT_GPU_STATIC_LIBS "Link to static GPU libraries"          ${AFFT_STATIC_LIBS})
option(AFFT_MODULE          "Enable C++20 module"                                     OFF)

set(AFFT_MAX_DIM_COUNT 4                 CACHE STRING "Maximum number of dimensions supported by the library, default is 4")
set(AFFT_BACKEND_LIST  "POCKETFFT;VKFFT" CACHE STRING "Semicolon separated list of backends to use, default is POCKETFFT and VKFFT")
set(AFFT_GPU_BACKEND   ""                CACHE STRING "GPU framework to use (CUDA, HIP or OPENCL), default is none (no GPU support)")
set(AFFT_MP_BACKEND    ""                CACHE STRING "Multi process framework to use (MPI), default is none (no MP support)")

if(NOT (CMAKE_VERSION VERSION_GREATER_EQUAL 3.28) AND AFFT_MODULE)
  message(FATAL_ERROR "C++20 module support requires CMake 3.28 or later")
endif()

set(CMAKE_CXX_SCAN_FOR_MODULES ON)

########################################################################################################################
# Create targets
########################################################################################################################
add_library(afft ${CMAKE_CURRENT_SOURCE_DIR}/src/afft.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/src/alloc.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/src/backend.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/src/init.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/src/Plan.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp)
add_library(afft::afft ALIAS afft)
add_library(afft-header-only INTERFACE)
add_library(afft::afft-header-only ALIAS afft-header-only)

if(AFFT_MODULE)
  add_library(afft-module)
  add_library(afft::afft-module ALIAS afft-module)

  target_sources(afft-module PUBLIC FILE_SET CXX_MODULES FILES ${CMAKE_CURRENT_SOURCE_DIR}/mod/afft.cpp)
endif()

target_compile_features(afft PUBLIC cxx_std_17)
target_include_directories(afft PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(afft SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/span/include)

target_compile_features(afft-header-only INTERFACE cxx_std_17)
target_include_directories(afft-header-only INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(afft-header-only SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/span/include)
target_compile_definitions(afft-header-only INTERFACE AFFT_HEADER_ONLY)

if (TARGET afft-module)
  target_compile_features(afft-module PUBLIC cxx_std_20)
  target_include_directories(afft-module PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_include_directories(afft-module SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/span/include)
endif()

########################################################################################################################
# Set up GPU target if needed
########################################################################################################################
if(NOT ("${AFFT_GPU_BACKEND}" STREQUAL ""))
  string(TOUPPER "${AFFT_GPU_BACKEND}" AFFT_GPU_BACKEND)

  set(GPU_LIBRARIES "")

  if(AFFT_GPU_BACKEND STREQUAL "CUDA")
    set(AFFT_ENABLE_CUDA TRUE)
    if(AFFT_USE_NVHPC_CUDA)
      find_package(NVHPC REQUIRED)
      find_package(CUDAToolkit REQUIRED COMPONENTS cuda_driver)
      list(APPEND GPU_LIBRARIES CUDA::cuda_driver NVHPC::CUDART NVHPC::NVRTC)
      set(AFFT_CUDA_ROOT_DIR "${NVHPC_ROOT_DIR}/cuda")
    else()
      find_package(CUDAToolkit REQUIRED)
      list(APPEND GPU_LIBRARIES CUDA::cuda_driver CUDA::cudart CUDA::nvrtc)
      cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH AFFT_CUDA_ROOT_DIR)
    endif()
  elseif(AFFT_GPU_BACKEND STREQUAL "HIP")
    set(AFFT_ENABLE_HIP TRUE)
    find_package(hip REQUIRED)
    list(APPEND GPU_LIBRARIES hip::host)
    set(AFFT_HIP_ROOT_DIR="${HIP_ROOT_DIR}")
    if((HIP_PLATFORM STREQUAL "amd") OR (HIP_PLATFORM STREQUAL "hcc"))
    elseif((HIP_PLATFORM STREQUAL "nvidia") OR (HIP_PLATFORM STREQUAL "nvcc"))
      find_package(CUDAToolkit REQUIRED)
      list(APPEND GPU_LIBRARIES CUDA::cuda_driver CUDA::cudart CUDA::nvrtc)
    else()
      message(FATAL_ERROR "Invalid HIP platform: ${HIP_PLATFORM}")
    endif()
  elseif(AFFT_GPU_BACKEND STREQUAL "OPENCL")
    set(AFFT_ENABLE_OPENCL TRUE)
    if(AFFT_USE_CUDA_OPENCL)
      find_package(CUDAToolkit REQUIRED COMPONENTS OpenCL)
      list(APPEND GPU_LIBRARIES CUDA::OpenCL)
    else()
      find_package(OpenCL REQUIRED)
      list(APPEND GPU_LIBRARIES OpenCL::OpenCL)
    endif()
  else()
    message(FATAL_ERROR "Invalid GPU backend: ${AFFT_GPU_BACKEND}")
  endif()

  target_link_libraries(afft PUBLIC ${GPU_LIBRARIES})
  target_link_libraries(afft-header-only INTERFACE ${GPU_LIBRARIES})
  if(TARGET afft-module)
    target_link_libraries(afft-module PUBLIC ${GPU_LIBRARIES})
  endif()
endif()

########################################################################################################################
# Set up MP target if needed
########################################################################################################################
if(NOT ("${AFFT_MP_BACKEND}" STREQUAL ""))
  set(HAS_MP_SUPPORT TRUE)
  string(TOUPPER "${AFFT_MP_BACKEND}" AFFT_MP_BACKEND)

  set(MP_LIBRARIES "")

  if(AFFT_MP_BACKEND STREQUAL "MPI")
    set(AFFT_ENABLE_MPI TRUE)
    if(AFFT_USE_NVHPC_MPI)
      find_package(NVHPC REQUIRED)
      list(APPEND MP_LIBRARIES NVHPC::MPI)
    else()
      find_package(MPI REQUIRED)
      list(APPEND MP_LIBRARIES MPI::MPI_CXX)
    endif()
  else()
    message(FATAL_ERROR "Invalid multi-process backend: ${AFFT_MP_BACKEND}")
  endif()

  target_link_libraries(afft PUBLIC ${MP_LIBRARIES})
  target_link_libraries(afft-header-only INTERFACE ${MP_LIBRARIES})
  if(TARGET afft-module)
    target_link_libraries(afft-module PUBLIC ${MP_LIBRARIES})
  endif()
else()
  set(HAS_MP_SUPPORT FALSE)
endif()

########################################################################################################################
# Set up backend libraries
########################################################################################################################
# Convert the list of backends to uppercase and pass it to the target
string(TOUPPER "${AFFT_BACKEND_LIST}" BACKEND_LIST_UPPER)
list(TRANSFORM BACKEND_LIST_UPPER PREPEND "AFFT_BACKEND_" OUTPUT_VARIABLE BACKEND_LIST_UPPER_PREP)
string(REPLACE ";" "|" AFFT_BACKEND_MASK "${BACKEND_LIST_UPPER_PREP}")

set(BACKEND_LIBRARIES "")
set(BACKEND_INCLUDES  "")

foreach(BACKEND ${AFFT_BACKEND_LIST})
  if(BACKEND STREQUAL "CLFFT")
    set(AFFT_ENABLE_CLFFT TRUE)
    if(AFFT_GPU_BACKEND STREQUAL "OPENCL")
      status(CHECK_START "Finding clFFT library")
      find_library(CLFFT_LIB NAMES "libclFFT" "clFFT" REQUIRED)
      reportFound(CLFFT_LIB)

      list(APPEND BACKEND_LIBRARIES ${CLFFT_LIB})
    endif()
  elseif(BACKEND STREQUAL "CUFFT")
    set(AFFT_ENABLE_CUFFT TRUE)
    if(AFFT_GPU_BACKEND STREQUAL "CUDA")
      if (AFFT_USE_NVHPC_CUFFT)
        find_package(NVHPC REQUIRED)
        if(AFFT_ENABLE_CUFFTMP)
          if(HAS_MP_SUPPORT AND AFFT_MP_BACKEND STREQUAL "MPI")
            set(AFFT_CUFFT_HAS_MP TRUE)
            list(APPEND BACKEND_LIBRARIES NVHPC::CUFFTMP)
          else()
            message(WARNING "cufftMp requires MPI support. Disabling cufftMp.")
          endif()
        else()
          list(APPEND BACKEND_LIBRARIES NVHPC::CUFFT)
        endif()
      else()
        list(APPEND BACKEND_LIBRARIES CUDA::cufft)
      endif()
    else()
      message(WARNING "cuFFT requires CUDA GPU backend. Disabling cuFFT.")
    endif()
  elseif(BACKEND STREQUAL "FFTW3")
    set(AFFT_ENABLE_FFTW3 TRUE)
    message(CHECK_START "Finding fftw3 library")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")

    message(CHECK_START "Finding fftw3f library")
    find_library(FFTW3F_LIB NAMES "libfftw3f" "fftw3f")
    reportFound(FFTW3F_LIB)
    
    message(CHECK_START "Finding fftw3 library")
    find_library(FFTW3_LIB NAMES "libfftw3" "fftw3")
    reportFound(FFTW3_LIB)

    message(CHECK_START "Finding fftw3l library")
    find_library(FFTW3L_LIB NAMES "libfftw3l" "fftw3l")
    reportFound(FFTW3L_LIB)

    message(CHECK_START "Finding fftw3q library")
    find_library(FFTW3Q_LIB NAMES "libfftw3q" "fftw3q")
    reportFound(FFTW3Q_LIB)

    if(FFTW3F_LIB)
      set(AFFT_FFTW3_HAS_FLOAT TRUE)
      list(APPEND BACKEND_LIBRARIES ${FFTW3F_LIB})
    endif()
    if (FFTW3_LIB)
      set(AFFT_FFTW3_HAS_DOUBLE TRUE)
      list(APPEND BACKEND_LIBRARIES ${FFTW3_LIB})
    endif()
    if (FFTW3L_LIB)
      set(AFFT_FFTW3_HAS_LONG TRUE)
      list(APPEND BACKEND_LIBRARIES ${FFTW3L_LIB})
    endif()
    if (FFTW3Q_LIB)
      set(AFFT_FFTW3_HAS_QUAD TRUE)
      list(APPEND BACKEND_LIBRARIES ${FFTW3Q_LIB})
    endif()

    message(CHECK_START "Finding fftw3f_threads library")
    find_library(FFTW3F_THREADS_LIB NAMES "libfftw3f_threads" "fftw3f_threads")
    reportFound(FFTW3F_THREADS_LIB)

    message(CHECK_START "Finding fftw3_threads library")
    find_library(FFTW3_THREADS_LIB NAMES "libfftw3_threads" "fftw3_threads")
    reportFound(FFTW3_THREADS_LIB)

    message(CHECK_START "Finding fftw3l_threads library")
    find_library(FFTW3L_THREADS_LIB NAMES "libfftw3l_threads" "fftw3l_threads")
    reportFound(FFTW3L_THREADS_LIB)

    message(CHECK_START "Finding fftw3q_threads library")
    find_library(FFTW3Q_THREADS_LIB NAMES "libfftw3q_threads" "fftw3q_threads")
    reportFound(FFTW3Q_THREADS_LIB)

    if(FFTW3F_THREADS_LIB)
      set(AFFT_FFTW3_HAS_FLOAT_THREADS TRUE)
      list(APPEND BACKEND_LIBRARIES ${FFTW3F_THREADS_LIB})
    endif()
    if (FFTW3_THREADS_LIB)
      set(AFFT_FFTW3_HAS_DOUBLE_THREADS TRUE)
      list(APPEND BACKEND_LIBRARIES ${FFTW3_THREADS_LIB})
    endif()
    if (FFTW3L_THREADS_LIB)
      set(AFFT_FFTW3_HAS_LONG_THREADS TRUE)
      list(APPEND BACKEND_LIBRARIES ${FFTW3L_THREADS_LIB})
    endif()
    if (FFTW3Q_THREADS_LIB)
      set(AFFT_FFTW3_HAS_QUAD_THREADS TRUE)
      list(APPEND BACKEND_LIBRARIES ${FFTW3Q_THREADS_LIB})
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
        set(AFFT_FFTW3_HAS_MPI_FLOAT TRUE)
        list(APPEND BACKEND_LIBRARIES ${FFTW3F_MPI_LIB})
      endif()
      if (FFTW3_LIB AND FFTW3_THREADS_LIB AND FFTW3F_MPI_LIB)
        set(AFFT_FFTW3_HAS_MPI_DOUBLE TRUE)
        list(APPEND BACKEND_LIBRARIES ${FFTW3_MPI_LIB})
      endif()
      if (FFTW3L_LIB AND FFTW3L_THREADS_LIB AND FFTW3F_MPI_LIB)
        set(AFFT_FFTW3_HAS_MPI_LONG TRUE)
        list(APPEND BACKEND_LIBRARIES ${FFTW3L_MPI_LIB})
      endif()
    endif()

    list(POP_BACK CMAKE_MESSAGE_INDENT)
  elseif(BACKEND STREQUAL "MKL")
    set(AFFT_ENABLE_MKL TRUE)
    find_package(MKL REQUIRED)
    list(APPEND BACKEND_LIBRARIES MKL::MKL)

    # TODO: MPI support
  elseif(BACKEND STREQUAL "POCKETFFT")
    set(AFFT_ENABLE_POCKETFFT TRUE)
    list(APPEND BACKEND_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pocketfft)
  elseif(BACKEND STREQUAL "HIPFFT")
    if(AFFT_GPU_BACKEND STREQUAL "HIP")
      find_package(hipfft REQUIRED)
      list(APPEND BACKEND_LIBRARIES hip::hipfft)
    else()
      message(WARNING "hipFFT requires HIP GPU backend. Disabling hipFFT.")
    endif()
  elseif(BACKEND STREQUAL "ROCFFT")
    set(AFFT_ENABLE_ROCFFT TRUE)
    if(AFFT_GPU_BACKEND STREQUAL "HIP")
      find_package(rocfft REQUIRED)
      list(APPEND BACKEND_LIBRARIES roc::rocfft)
    else()
      message(WARNING "rocFFT requires HIP GPU backend. Disabling rocFFT.")
    endif()
  elseif(BACKEND STREQUAL "VKFFT")
    if((AFFT_GPU_BACKEND STREQUAL "CUDA") OR (AFFT_GPU_BACKEND STREQUAL "HIP") OR (AFFT_GPU_BACKEND STREQUAL "OPENCL"))
      set(AFFT_ENABLE_VKFFT TRUE)
      list(APPEND BACKEND_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/VkFFT/vkFFT)

      message(CHECK_START "Finding quadmath library")
      find_library(QUADMATH_LIB NAMES "quadmath")
      reportFound(QUADMATH_LIB)

      if(QUADMATH_LIB)
        set(AFFT_VKFFT_HAS_DOUBLE_DOUBLE TRUE)
        list(APPEND BACKEND_LIBRARIES ${QUADMATH_LIB})
      endif()
    else()
      message(WARNING "VKFFT requires CUDA, HIP or OPENCL GPU backend. Disabling VKFFT.")
    endif()
  else()
    message(FATAL_ERROR "Invalid backend: ${BACKEND}")
  endif()
endforeach()

target_link_libraries(afft PRIVATE ${BACKEND_LIBRARIES})
target_include_directories(afft SYSTEM PRIVATE ${BACKEND_INCLUDES})

target_link_libraries(afft-header-only INTERFACE ${BACKEND_LIBRARIES})
target_include_directories(afft-header-only SYSTEM INTERFACE ${BACKEND_INCLUDES})

if(TARGET afft-module)
  target_link_libraries(afft-module PRIVATE ${BACKEND_LIBRARIES})
  target_include_directories(afft-module SYSTEM PRIVATE ${BACKEND_INCLUDES})
endif()

########################################################################################################################
# Make the library configuration files
########################################################################################################################
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/afft/afft-version.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/include/afft/afft-version.h)
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/afft/afft-config.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/include/afft/afft-config.h)

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

  add_executable(cpu_c_dft_3D_C2C_transpose examples/cpu/dft_3D_C2C_transpose.c)
  set_target_properties(cpu_c_dft_3D_C2C_transpose PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/examples")
  target_link_libraries(cpu_c_dft_3D_C2C_transpose PRIVATE afft::afft)

  if(AFFT_MODULE)
    add_executable(cpu_dft_1D_C2C_simple_mod examples/cpu/dft_1D_C2C_simple_mod.cpp)
    set_target_properties(cpu_dft_1D_C2C_simple_mod PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/examples")
    target_compile_features(cpu_dft_1D_C2C_simple_mod PRIVATE cxx_std_20)
    target_link_libraries(cpu_dft_1D_C2C_simple_mod PRIVATE afft::afft-module)
  endif()

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

    add_executable(gpu_dft_3D_C2C_transpose "examples/gpu/${GPU_EXAMPLES_FRAMEWORK_DIR}/dft_3D_C2C_transpose.c")
    set_target_properties(gpu_dft_3D_C2C_transpose PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/examples")
    target_link_libraries(gpu_dft_3D_C2C_transpose PRIVATE afft::afft)
  endif()
endif()

# Should be implemented as a function over all files in the source directory
if(AFFT_BUILD_TESTS)
  
endif()
