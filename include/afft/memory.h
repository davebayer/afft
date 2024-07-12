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

#ifndef AFFT_MEMORY_H
#define AFFT_MEMORY_H

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.h"
#endif

#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Alignment type
typedef size_t afft_Alignment;

/// @brief Alignment enumeration
enum
{
  afft_Alignment_simd128  = 16,  ///< 128-bit SIMD alignment
  afft_Alignment_simd256  = 32,  ///< 256-bit SIMD alignment
  afft_Alignment_simd512  = 64,  ///< 512-bit SIMD alignment
  afft_Alignment_simd1024 = 128, ///< 1024-bit SIMD alignment
  afft_Alignment_simd2048 = 256, ///< 2048-bit SIMD alignment

  afft_Alignment_sse    = afft_Alignment_simd128,  ///< SSE alignment
  afft_Alignment_sse2   = afft_Alignment_simd128,  ///< SSE2 alignment
  afft_Alignment_sse3   = afft_Alignment_simd128,  ///< SSE3 alignment
  afft_Alignment_sse4   = afft_Alignment_simd128,  ///< SSE4 alignment
  afft_Alignment_sse4_1 = afft_Alignment_simd128,  ///< SSE4.1 alignment
  afft_Alignment_sse4_2 = afft_Alignment_simd128,  ///< SSE4.2 alignment
  afft_Alignment_avx    = afft_Alignment_simd256,  ///< AVX alignment
  afft_Alignment_avx2   = afft_Alignment_simd256,  ///< AVX2 alignment
  afft_Alignment_avx512 = afft_Alignment_simd512,  ///< AVX-512 alignment
  afft_Alignment_neon   = afft_Alignment_simd128,  ///< NEON alignment
  afft_Alignment_sve    = afft_Alignment_simd2048, ///< SVE alignment
};

/// @brief Memory layout of the centralized transform.
typedef struct
{
  afft_ComplexFormat complexFormat; ///< Complex number format
  const afft_Size*   srcStrides;    ///< Source strides
  afft_Alignment     srcAlignment;  ///< Source alignment
  const afft_Size*   dstStrides;    ///< Destination strides
  afft_Alignment     dstAlignment;  ///< Destination alignment      
} afft_MemoryLayout;

/// @brief Memory block
typedef struct
{
  const afft_Size* starts;    ///< Start indices
  const afft_Size* sizes;     ///< Sizes
  const afft_Size* strides;   ///< Strides
  afft_Alignment   alignment; ///< Alignment
} afft_MemoryBlock;

/// @brief Memory layout of the distributed transform.
typedef struct
{
  afft_ComplexFormat      complexFormat;  ///< Complex number format
  const afft_MemoryBlock* srcBlocks;      ///< Source blocks
  const afft_Axis*        srcDistribAxes; ///< Source distributed axes
  const afft_MemoryBlock* dstBlocks;      ///< Destination blocks
  const afft_Axis*        dstDistribAxes; ///< Destination distributed axes
} afft_DistribMemoryLayout;

/**
 * @brief Allocate aligned memory.
 * @param sizeInBytes Size of the memory block in bytes.
 * @param alignment Alignment of the memory block.
 * @return Pointer to the allocated memory block or NULL if the allocation failed.
 */
void* afft_cpu_alignedAlloc(size_t sizeInBytes, afft_Alignment alignment);

/**
 * @brief Free aligned memory.
 * @param ptr Pointer to the memory block.
 * @param alignment Alignment of the memory block.
 */
void afft_cpu_alignedFree(void* ptr, afft_Alignment alignment);

#ifdef __cplusplus
}
#endif

#endif /* AFFT_MEMORY_H */
