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

if(AFFT_VKFFT_BACKEND STREQUAL "CUDA")
  set(AFFT_VKFFT_BACKEND_NUM 1)
elseif(AFFT_VKFFT_BACKEND STREQUAL "HIP")
  set(AFFT_VKFFT_BACKEND_NUM 2)
elseif(AFFT_VKFFT_BACKEND STREQUAL "OPENCL")
  set(AFFT_VKFFT_BACKEND_NUM 3)
elseif("${AFFT_VKFFT_BACKEND}" STREQUAL "")
  unset(AFFT_VKFFT_BACKEND_NUM)
else()
  message(FATAL_ERROR "Invalid VKFFT backend: ${AFFT_VKFFT_BACKEND}")
endif()  

########################################################################################################################
# Set up targets
########################################################################################################################
set(TARGET_LIBRARIES "")

foreach(TARGET ${AFFT_TARGET_LIST})
  if(${TARGET} STREQUAL "CPU")
    set(AFFT_ENABLE_CPU TRUE)
  elseif(${TARGET} STREQUAL "CUDA")
    set(AFFT_ENABLE_CUDA TRUE)
    if(AFFT_USE_NVHPC_CUDA)
      find_package(NVHPC REQUIRED)
      find_package(CUDAToolkit REQUIRED COMPONENTS cuda_driver)
      list(APPEND TARGET_LIBRARIES CUDA::cuda_driver NVHPC::CUDART NVHPC::NVRTC)
      set(AFFT_CUDA_ROOT_DIR "${NVHPC_ROOT_DIR}/cuda")
    else()
      find_package(CUDAToolkit REQUIRED)
      list(APPEND TARGET_LIBRARIES CUDA::cuda_driver CUDA::cudart CUDA::nvrtc)
      cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH AFFT_CUDA_ROOT_DIR)
    endif()
  elseif(${TARGET} STREQUAL "HIP")
    set(AFFT_ENABLE_HIP TRUE)
    find_package(hip REQUIRED)
    list(APPEND TARGET_LIBRARIES hip::host)
    if((HIP_PLATFORM STREQUAL "amd") OR (HIP_PLATFORM STREQUAL "hcc"))
    elseif((HIP_PLATFORM STREQUAL "nvidia") OR (HIP_PLATFORM STREQUAL "nvcc"))
      find_package(CUDAToolkit REQUIRED)
      list(APPEND TARGET_LIBRARIES CUDA::cuda_driver CUDA::cudart CUDA::nvrtc)
    else()
      message(FATAL_ERROR "Invalid HIP platform: ${HIP_PLATFORM}")
    endif()
  elseif(${TARGET} STREQUAL "OPENCL")
    set(AFFT_ENABLE_OPENCL TRUE)
    if(AFFT_USE_CUDA_OPENCL)
      find_package(CUDAToolkit REQUIRED COMPONENTS OpenCL)
      list(APPEND TARGET_LIBRARIES CUDA::OpenCL)
    else()
      find_package(OpenCL REQUIRED)
      list(APPEND TARGET_LIBRARIES OpenCL::OpenCL)
    endif()
  elseif(${TARGET} STREQUAL OPENMP)
    set(AFFT_ENABLE_OPENMP TRUE)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    list(APPEND TARGET_LIBRARIES OpenMP::OpenMP_CXX)
  else()
    message(FATAL_ERROR "Invalid target: ${TARGET}")
  endif()
endforeach()

########################################################################################################################
# Set up MP target if needed
########################################################################################################################
if(NOT ("${AFFT_MP_BACKEND}" STREQUAL ""))
  set(HAS_MP_SUPPORT TRUE)

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
else()
  set(HAS_MP_SUPPORT FALSE)
endif()

########################################################################################################################
# Set up backend libraries
########################################################################################################################
set(BACKEND_LIBRARIES "")
set(BACKEND_INCLUDES  "")

foreach(BACKEND ${AFFT_BACKEND_LIST})
  if(BACKEND STREQUAL "CLFFT")
    set(AFFT_ENABLE_CLFFT TRUE)
    if(AFFT_ENABLE_OPENCL)
      status(CHECK_START "Finding clFFT library")
      find_library(CLFFT_LIB NAMES "libclFFT" "clFFT" REQUIRED)
      reportFound(CLFFT_LIB)
      list(APPEND BACKEND_LIBRARIES ${CLFFT_LIB})
    endif()
  elseif(BACKEND STREQUAL "CUFFT")
    set(AFFT_ENABLE_CUFFT TRUE)
    if(AFFT_ENABLE_CUDA)
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
      message(WARNING "cuFFT requires CUDA support. Disabling cuFFT.")
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
    list(APPEND BACKEND_INCLUDES ${PROJECT_SOURCE_DIR}/3rdparty/pocketfft)
  elseif(BACKEND STREQUAL "HIPFFT")
    if(AFFT_ENABLE_HIP)
      find_package(hipfft REQUIRED)
      list(APPEND BACKEND_LIBRARIES hip::hipfft)
    else()
      message(WARNING "hipFFT requires HIP support. Disabling hipFFT.")
    endif()
  elseif(BACKEND STREQUAL "ROCFFT")
    set(AFFT_ENABLE_ROCFFT TRUE)
    if(AFFT_ENABLE_HIP)
      find_package(rocfft REQUIRED)
      list(APPEND BACKEND_LIBRARIES roc::rocfft)
    else()
      message(WARNING "rocFFT requires HIP support. Disabling rocFFT.")
    endif()
  elseif(BACKEND STREQUAL "VKFFT")
    if(AFFT_ENABLE_CUDA OR AFFT_ENABLE_HIP OR AFFT_ENABLE_OPENCL)
      set(AFFT_ENABLE_VKFFT TRUE)
      list(APPEND BACKEND_INCLUDES ${PROJECT_SOURCE_DIR}/3rdparty/VkFFT/vkFFT)

      message(CHECK_START "Finding quadmath library")
      find_library(QUADMATH_LIB NAMES "quadmath")
      reportFound(QUADMATH_LIB)

      if(QUADMATH_LIB)
        set(AFFT_VKFFT_HAS_DOUBLE_DOUBLE TRUE)
        list(APPEND BACKEND_LIBRARIES ${QUADMATH_LIB})
      endif()
    else()
      message(WARNING "VKFFT requires CUDA, HIP or OPENCL support. Disabling VKFFT.")
    endif()
  else()
    message(FATAL_ERROR "Invalid backend: ${BACKEND}")
  endif()
endforeach()

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

########################################################################################################################
# Create targets
########################################################################################################################
add_subdirectory(src)
add_library(afft::afft ALIAS afft)
add_library(afft::afft-header-only ALIAS afft-header-only)

target_link_libraries(afft
  PUBLIC  ${MP_LIBRARIES} ${TARGET_LIBRARIES}
  PRIVATE ${BACKEND_LIBRARIES})
target_include_directories(afft SYSTEM
  PRIVATE ${BACKEND_INCLUDES})

target_link_libraries(afft-header-only
  INTERFACE ${MP_LIBRARIES} ${TARGET_LIBRARIES} ${BACKEND_LIBRARIES})
target_include_directories(afft-header-only SYSTEM
  INTERFACE ${BACKEND_INCLUDES})

if(AFFT_BUILD_MODULE)
  add_subdirectory(mod)
  add_library(afft::afft-module ALIAS afft-module)

  target_link_libraries(afft-module
    PUBLIC  ${MP_LIBRARIES} ${TARGET_LIBRARIES}
    PRIVATE ${BACKEND_LIBRARIES})
  target_include_directories(afft-module SYSTEM
    PRIVATE ${BACKEND_INCLUDES})
endif()

########################################################################################################################
# Add examples if requested
########################################################################################################################
if(AFFT_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

########################################################################################################################
# Add tests if requested
########################################################################################################################
if(AFFT_BUILD_TESTS)
  add_subdirectory(tests)
endif()

########################################################################################################################
# Install
########################################################################################################################
