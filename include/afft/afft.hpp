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

#include <version>

#ifndef __cpp_lib_concepts
# error "C++20 concepts are required"
#endif

#ifndef __cpp_lib_format
# if !__has_include(<fmt/format.h>)
#   error "fmtlib or C++20 std::format is required"
# endif
#endif

#ifndef __cpp_lib_integer_comparison_functions
# error "C++20 integer comparison functions are required"
#endif

#ifndef AFFT_MAX_DIM_COUNT
# define AFFT_MAX_DIM_COUNT                      4
# define AFFT_UNDEF_MAX_DIM_COUNT
#endif

// If CPU backend list is not defined, use PocketFFT
#ifndef AFFT_CPU_BACKEND_LIST
# define AFFT_CPU_BACKEND_LIST                   POCKETFFT
# define AFFT_UNDEF_CPU_BACKEND_LIST
#endif

// If GPU backend list is not defined, but gpu framework is selected, use VkFFT
#ifdef AFFT_GPU_FRAMEWORK
# ifndef AFFT_GPU_BACKEND_LIST
#   define AFFT_GPU_BACKEND_LIST                 VKFFT
#   define AFFT_UNDEF_GPU_BACKEND_LIST
# endif
#endif

#define AFFT_VERSION_MAJOR                       0
#define AFFT_VERSION_MINOR                       1
#define AFFT_VERSION_PATCH                       0

#include "cpu.hpp"
#include "gpu.hpp"
#include "common.hpp"
#include "init.hpp"
#include "type.hpp"
#include "Plan.hpp"
// #include "PlanCache.hpp" // future feature
#include "utils.hpp"

namespace afft
{
  /**
   * @struct Version
   * @brief AFFT version.
   */
  struct Version
  {
    static constexpr int major{AFFT_VERSION_MAJOR}; ///< Major version.
    static constexpr int minor{AFFT_VERSION_MINOR}; ///< Minor version.
    static constexpr int patch{AFFT_VERSION_PATCH}; ///< Patch version.
  } version;
} // namespace afft

#ifdef AFFT_UNDEF_GPU_BACKEND_LIST
# undef AFFT_GPU_BACKEND_LIST
# undef AFFT_UNDEF_GPU_BACKEND_LIST
#endif

#ifdef AFFT_UNDEF_CPU_BACKEND_LIST
# undef AFFT_CPU_BACKEND_LIST
# undef AFFT_UNDEF_CPU_BACKEND_LIST
#endif

#ifdef AFFT_UNDEF_MAX_DIM_COUNT
# undef AFFT_MAX_DIM_COUNT
# undef AFFT_UNDEF_MAX_DIM_COUNT
#endif

#endif /* AFFT_AFFT_HPP */
