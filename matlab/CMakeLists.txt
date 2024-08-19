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

cmake_minimum_required(VERSION 3.22)

find_package(Matlab REQUIRED COMPONENTS MAIN_PROGRAM MEX_COMPILER)

# If CUDA is enabled, enable GPU support in mex-cpp-wrapper
list(FIND AFFT_TARGETS "CUDA" CUDA_TARGET_INDEX)
if(CUDA_TARGET_INDEX GREATER -1)
  set(MEXW_ENABLE_GPU ON)
endif()

# Add mex-cpp-wrapper
add_subdirectory(3rdparty/mex-cpp-wrapper)

if(MEXW_ENABLE_GPU)
  set(LINK_TARGETS afft::afft-header-only mexw::mexw-gpu)
else()
  set(LINK_TARGETS afft::afft-header-only mexw::mexw)
endif()

set(MEX_SOURCES src/mex/afft.cpp)

matlab_add_mex(
  NAME          afft_matlab_mex
  SRC           ${MEX_SOURCES}
  OUTPUT_NAME   afft
  LINK_TO       ${LINK_TARGETS}
  R2018a
)
add_custom_command(TARGET afft_matlab_mex PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/package ${CMAKE_CURRENT_BINARY_DIR}/package
)
set_property(TARGET afft_matlab_mex
  PROPERTY LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/package/+afft/private)
