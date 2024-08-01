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

if(NOT (CMAKE_VERSION VERSION_GREATER_EQUAL 3.28) AND AFFT_MODULE)
  message(FATAL_ERROR "C++20 module support requires CMake 3.28 or later")
endif()

add_library(afft-module)

set_target_properties(afft-module PROPERTIES CXX_SCAN_FOR_MODULES TRUE)

target_sources(afft-module
  PUBLIC FILE_SET CXX_MODULES
    FILES afft.cpp)

target_compile_features(afft-module PUBLIC cxx_std_20)
target_include_directories(afft-module PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(afft-module SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/span/include)
