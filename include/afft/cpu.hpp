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

#ifndef AFFT_CPU_HPP
#define AFFT_CPU_HPP

#include "macro.hpp"

/// @brief Macro for FFTW3 CPU transform backend
#define AFFT_CPU_TRANSFORM_BACKEND_FFTW3     (1 << 0)
/// @brief Macro for MKL CPU transform backend
#define AFFT_CPU_TRANSFORM_BACKEND_MKL       (1 << 1)
/// @brief Macro for PocketFFT CPU transform backend
#define AFFT_CPU_TRANSFORM_BACKEND_POCKETFFT (1 << 2)

/**
 * @brief Implementation of AFFT_CPU_TRANSFORM_BACKEND_FROM_NAME
 * @param backendName Name of the backend
 * @return Transform backend
 * @warning Do not use this macro directly
 */
#define AFFT_CPU_TRANSFORM_BACKEND_FROM_NAME_IMPL(backendName) \
  AFFT_CPU_TRANSFORM_BACKEND_##backendName

/**
 * @brief Macro for getting the transform backend from the name
 * @param backendName Name of the backend
 * @return Transform backend
 */
#define AFFT_CPU_TRANSFORM_BACKEND_FROM_NAME(backendName) \
  AFFT_CPU_TRANSFORM_BACKEND_FROM_NAME_IMPL(backendName)

/**
 * @brief Implementation of AFFT_CPU_TRANSFORM_BACKEND_MASK
 * @param ... List of transform backend names
 * @return Transform backend mask
 * @warning Do not use this macro directly
 */
#define AFFT_CPU_TRANSFORM_BACKEND_MASK_IMPL(...) \
  AFFT_BITOR(AFFT_FOR_EACH_WITH_DELIM(AFFT_CPU_TRANSFORM_BACKEND_FROM_NAME, AFFT_DELIM_COMMA, __VA_ARGS__))

/**
 * @brief Macro for getting the transform backend mask
 * @return Transform backend mask
 * @warning Requires AFFT_GPU_TRANSFORM_BACKEND_LIST to be defined
 */
#define AFFT_CPU_TRANSFORM_BACKEND_MASK \
  AFFT_CPU_TRANSFORM_BACKEND_MASK_IMPL(AFFT_CPU_TRANSFORM_BACKEND_LIST)

/**
 * @brief Macro for checking if the transform backend is allowed
 * @param backendName Name of the backend
 * @return Non zero if the transform backend is allowed, false otherwise
 */
#define AFFT_CPU_TRANSFORM_BACKEND_IS_ALLOWED(backendName) \
  (AFFT_CPU_TRANSFORM_BACKEND_FROM_NAME(backendName) & AFFT_CPU_TRANSFORM_BACKEND_MASK)

#include <cstddef>

#include "target.hpp"

namespace afft::cpu
{
  /// @brief Enumeration of CPU transform backends
  enum class TransformBackend
  {
    fftw3     = AFFT_CPU_TRANSFORM_BACKEND_FFTW3,
    mkl       = AFFT_CPU_TRANSFORM_BACKEND_MKL,
    pocketfft = AFFT_CPU_TRANSFORM_BACKEND_POCKETFFT,
  };

  /// @brief Alignment for CPU memory allocation
  enum class Alignment : std::size_t
  {
    defaultNew = __STDCPP_DEFAULT_NEW_ALIGNMENT__, ///< Default alignment for new operator
    simd128    = 16,                               ///< 128-bit SIMD alignment
    simd256    = 32,                               ///< 256-bit SIMD alignment
    simd512    = 64,                               ///< 512-bit SIMD alignment

    sse        = simd128,                          ///< SSE alignment
    sse2       = simd128,                          ///< SSE2 alignment
    sse3       = simd128,                          ///< SSE3 alignment
    sse4       = simd128,                          ///< SSE4 alignment
    sse4_1     = simd128,                          ///< SSE4.1 alignment
    sse4_2     = simd128,                          ///< SSE4.2 alignment
    avx        = simd256,                          ///< AVX alignment
    avx2       = simd256,                          ///< AVX2 alignment
    avx512     = simd512,                          ///< AVX-512 alignment
    neon       = simd128,                          ///< NEON alignment
  };
  /**
   * @struct Parameters
   * @brief Parameters for CPU transform
   */
  struct Parameters
  {
    Alignment alignment{Alignment::defaultNew}; ///< Alignment for CPU memory allocation, defaults to `Alignment::defaultNew`
    unsigned  threadLimit{0u};                  ///< Thread limit for CPU transform, 0 for no limit
  }; 
} // namespace afft::cpu

#endif /* AFFT_CPU_HPP */
