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

# General options
option(AFFT_BUILD_EXAMPLES  "Build examples"                                          OFF)
option(AFFT_BUILD_TESTS     "Build tests"                                             OFF)
option(AFFT_BUILD_MODULE    "Enable build of C++20 module"                            OFF)
option(AFFT_BUILD_MATLAB    "Build MATLAB bindings"                                   OFF)

# Packages lookup options
option(AFFT_USE_NVHPC_CUDA  "Use CUDA version that comes with NVHPC"                  OFF)
option(AFFT_USE_NVHPC_MPI   "Use MPI version that comes with NVHPC"                   OFF)
option(AFFT_ENABLE_CUFFTMP  "Enable multi-process support for cuFFT (requires NVHPC)" OFF)
option(AFFT_USE_NVHPC_CUFFT "Use cuFFT version that comes with NVHPC"                 ${AFFT_ENABLE_CUFFTMP})
option(AFFT_USE_CUDA_OPENCL "Use OpenCL version that comes with CUDA"                 OFF)

# Library configuration
set(AFFT_MAX_DIM_COUNT 4           CACHE STRING "Maximum number of dimensions supported by the library, default is 4")
set(AFFT_BACKENDS      "POCKETFFT" CACHE STRING "Semicolon separated list of backends to use, default is POCKETFFT and VKFFT")
set(AFFT_TARGETS       "CPU"       CACHE STRING "Semicolon separated list of targets to build for, default is CPU")
set(AFFT_MP_BACKEND    ""          CACHE STRING "Multi process framework to use (MPI), default is none (no MP support)")
set(AFFT_VKFFT_BACKEND ""          CACHE STRING "Which VkFFT backend to use (CUDA, HIP or OpenCL), default is empty (set to first available)")

# Includes
include(cmake/afftUtils.cmake)

# Enable testing if requested
if(AFFT_BUILD_TESTS)
  include(CTest)
  enable_testing()
endif()

# Add library subdirectory
add_subdirectory(library)

# Add matlab subdirectory
if(AFFT_BUILD_MATLAB)
  add_subdirectory(matlab)
endif()
