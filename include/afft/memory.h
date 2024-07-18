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

/// @brief Memory layout type
typedef uint8_t afft_MemoryLayout;

/// @brief Memory layout enumeration
enum
{
  afft_MemoryLayout_centralized, ///< Centralized memory layout, only when the transformation is executed by single process on single target
  afft_MemoryLayout_distributed, ///< Distributed memory layout, for distributed transformations over multiple processes or targets
};

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
  
  /// @brief native alignment
#if defined(__AVX512F__)
  afft_Alignment_cpuNative = afft_Alignment_avx512;
#elif defined(__AVX2__)
  afft_Alignment_cpuNative = afft_Alignment_avx2;
#elif defined(__AVX__)
  afft_Alignment_cpuNative = afft_Alignment_avx;
#elif defined(__SSE4_2__)
  afft_Alignment_cpuNative = afft_Alignment_sse4_2;
#elif defined(__SSE4_1__)
  afft_Alignment_cpuNative = afft_Alignment_sse4_1;
#elif defined(__SSE4__)
  afft_Alignment_cpuNative = afft_Alignment_sse4;
#elif defined(__SSE3__)
  afft_Alignment_cpuNative = afft_Alignment_sse3;
#elif defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP == 2)
  afft_Alignment_cpuNative = afft_Alignment_sse2;
#elif defined(__SSE__) || (defined(_M_IX86_FP) && _M_IX86_FP == 1)
  afft_Alignment_cpuNative = afft_Alignment_sse;
#elif defined(__ARM_NEON) || defined(_M_ARM_NEON)
  afft_Alignment_cpuNative = afft_Alignment_neon;
#elif (defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE == 1) || (defined(__ARM_FEATURE_SVE2) && __ARM_FEATURE_SVE2 == 1)
  afft_Alignment_cpuNative = afft_Alignment_sve;
#else
  afft_Alignment_cpuNative = afft_Alignment_simd128;
#endif
};

/// @brief Complex format type
typedef uint8_t afft_ComplexFormat;

/// @brief Complex format enumeration
enum
{
  afft_ComplexFormat_interleaved, ///< Interleaved
  afft_ComplexFormat_planar       ///< Planar
};

/// @brief Centralized memory layout structure
typedef struct afft_CentralizedMemoryLayout afft_CentralizedMemoryLayout;

/// @brief Memory block structure
typedef struct afft_MemoryBlock afft_MemoryBlock;

/// @brief Distributed memory layout structure
typedef struct afft_DistributedMemoryLayout afft_DistributedMemoryLayout;

/// @brief Centralized memory layout structure
struct afft_CentralizedMemoryLayout
{
  afft_Alignment     alignment;     ///< Memory alignment
  afft_ComplexFormat complexFormat; ///< Complex format
  const afft_Size*   srcStrides;    ///< Source strides (null for default or array of size shapeRank)
  const afft_Size*   dstStrides;    ///< Destination strides (null for default or array of size shapeRank)
};

/// @brief Memory block structure
struct afft_MemoryBlock
{
  const afft_Size* starts; ///< Start indices (null for default or array of size shapeRank)
  const afft_Size* sizes;  ///< Sizes (null for default or array of size shapeRank)
  const afft_Size* strides;///< Strides (null for default or array of size shapeRank)
};

/// @brief Distributed memory layout structure
struct afft_DistributedMemoryLayout
{
  afft_Alignment          alignment;      ///< Memory alignment
  afft_ComplexFormat      complexFormat;  ///< Complex format
  const afft_MemoryBlock* srcBlocks;      ///< Source blocks (null for default or array of size targetCount)
  const afft_Axis*        srcDistribAxes; ///< Source distribution axes (null for default or array of size targetCount)
  const afft_Axis*        srcAxesOrder;   ///< Source axes order (null for default or array of size shapeRank)
  const afft_MemoryBlock* dstBlocks;      ///< Destination blocks (null for default or array of size targetCount)
  const afft_Axis*        dstDistribAxes; ///< Destination distribution axes (null for default or array of size targetCount)
  const afft_Axis*        dstAxesOrder;   ///< Destination axes order (null for default or array of size shapeRank)
};

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
