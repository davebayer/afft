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

// If max dimension count is not defined, use 4 as default
#ifndef AFFT_MAX_DIM_COUNT
# define AFFT_MAX_DIM_COUNT                      0
# define AFFT_UNDEF_MAX_DIM_COUNT
#endif

// If distribution implementation mask is not defined, use 0 as default
#ifndef AFFT_DISTRIB_IMPL_MASK
# define AFFT_DISTRIB_IMPL_MASK                  0
# define AFFT_UNDEF_DISTRIB_IMPL_MASK
#endif

// If CPU backend mask is not defined, use PocketFFT
#ifndef AFFT_CPU_BACKEND_MASK
# define AFFT_CPU_BACKEND_MASK                   AFFT_CPU_BACKEND_POCKETFFT
# define AFFT_UNDEF_CPU_BACKEND_MASK
#endif

// If GPU backend list is not defined, but gpu framework is selected, use VkFFT
#ifdef AFFT_GPU_FRAMEWORK
# ifndef AFFT_GPU_BACKEND_MASK
#   define AFFT_GPU_BACKEND_MASK                 AFFT_GPU_BACKEND_VKFFT
#   define AFFT_UNDEF_GPU_BACKEND_MASK
# endif
#endif

#define AFFT_VERSION_MAJOR                       0 ///< Major version.
#define AFFT_VERSION_MINOR                       1 ///< Minor version.
#define AFFT_VERSION_PATCH                       0 ///< Patch version

#include "cpu.hpp"
#include "gpu.hpp"
#include "common.hpp"
#include "init.hpp"
#include "type.hpp"
#include "Plan.hpp"
#include "PlanCache.hpp"
#include "utils.hpp"

namespace afft
{
  /**
   * @struct Version
   * @brief AFFT version.
   */
  inline constexpr struct Version
  {
    static constexpr int major{AFFT_VERSION_MAJOR}; ///< Major version.
    static constexpr int minor{AFFT_VERSION_MINOR}; ///< Minor version.
    static constexpr int patch{AFFT_VERSION_PATCH}; ///< Patch version.
  } version;
} // namespace afft

#ifdef AFFT_UNDEF_GPU_BACKEND_MASK
# undef AFFT_GPU_BACKEND_MASK
# undef AFFT_UNDEF_GPU_BACKEND_MASK
#endif

#ifdef AFFT_UNDEF_CPU_BACKEND_MASK
# undef AFFT_CPU_BACKEND_MASK
# undef AFFT_UNDEF_CPU_BACKEND_MASK
#endif

#ifdef AFFT_UNDEF_DISTRIB_IMPL_MASK
# undef AFFT_DISTRIB_IMPL_MASK
# undef AFFT_UNDEF_DISTRIB_IMPL_MASK
#endif

#ifdef AFFT_UNDEF_MAX_DIM_COUNT
# undef AFFT_MAX_DIM_COUNT
# undef AFFT_UNDEF_MAX_DIM_COUNT
#endif

#endif /* AFFT_AFFT_HPP */
