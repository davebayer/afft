/*
  This file is part of afft library.

  Copyright (c) 2024 David Bayer

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

// Define the header only mode
#define AFFT_HEADER_ONLY

// Define the export macro.
#define AFFT_EXPORT export

// Global module fragment.
module;

// First include the C++ config header.
#include <afft/detail/config.hpp>

// Check the C++ version.
#if AFFT_CXX_VERSION < 202002L
# error "afft C++ module requires at least C++20"
#endif

// If import std is available, include only external backend headers.
#ifdef AFFT_CXX_HAS_IMPORT_STD
# define AFFT_INCLUDE_NO_STD
#endif

// Include all external headers.
#include <afft/detail/include.hpp>

// Define the module.
export module afft;

// Import the std module if available.
#ifdef AFFT_CXX_HAS_IMPORT_STD
import std;
#endif

// Include all public headers.
#include <afft/afft.hpp>
