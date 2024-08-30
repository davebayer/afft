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

#ifndef AFFT_AFFT_HPP
#define AFFT_AFFT_HPP

// Include the version header.
#include "afft-version.h"

// Include the config header.
#include "detail/config.hpp"

// Check for C++17 or later.
#if !(AFFT_CXX_VERSION > 201402L)
# error "afft C++ library requires C++17 or later. Please use the C api for older C++ versions."
#endif

// Include only once in the top-level header.
#include "detail/include.hpp"
#define AFFT_TOP_LEVEL_INCLUDE

// Check for std::uintptr_t.
#ifndef UINTPTR_MAX
# error "afft C++ library requires std::uintptr_t. Please use standard C++ library implementing std::uintptr_t."
#endif

// Include all public headers.
#include "backend.hpp"
#include "common.hpp"
#include "Description.hpp"
#include "error.hpp"
#include "fftw3.hpp"
#ifdef AFFT_CXX_HAS_FORMAT
# include "formatters.hpp"
#endif
#include "init.hpp"
#include "makePlan.hpp"
#include "memory.hpp"
#include "mp.hpp"
#include "Plan.hpp"
#include "PlanCache.hpp"
#include "select.hpp"
#include "Span.hpp"
#include "target.hpp"
#include "transform.hpp"
#include "type.hpp"
#include "typeTraits.hpp"
#include "utils.hpp"
#include "version.hpp"

#endif /* AFFT_AFFT_HPP */
