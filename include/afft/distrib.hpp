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

#ifndef AFFT_DISTRIB_HPP
#define AFFT_DISTRIB_HPP

/// @brief MPI distribution type
#define AFFT_DISTRIB_TYPE_MPI    (1 << 0)

/**
 * @brief Check if distribution type is enabled
 * @param typeName distribution type name
 * @return true if distribution type is enabled, false otherwise
 */
#define AFFT_DISTRIB_TYPE_IS_ENABLED(typeName) \
  (AFFT_DISTRIB_TYPE_MASK & AFFT_DISTRIB_TYPE_##typeName)

// Include distribution type headers
#if AFFT_DISTRIB_TYPE_IS_ENABLED(MPI)
# include <mpi.h>
#endif

namespace afft::distrib
{
  /**
   * @enum Type
   * @brief Distribution type
   */
  enum class Type
  {
    single, ///< single device, single process
    multi,  ///< multiple devices, single process
    mpi,    ///< mpi distribution, sigle device, multiple processes
  };

  /**
   * @struct MemoryBlock
   * @brief Memory block
   */
  struct MemoryBlock
  {
    View<std::size_t> starts{};  ///< starts of the memory block
    View<std::size_t> sizes{};   ///< sizes of the memory block
    View<std::size_t> strides{}; ///< strides of the memory block
  };

  /**
   * @struct MemoryLayout
   * @brief Memory layout
   */
  struct MemoryLayout
  {
    MemoryBlock srcBlock{};     ///< source memory block
    MemoryBlock dstBlock{};     ///< destination memory block
  };
} // namespace afft::distrib

#endif /* AFFT_DISTRIB_HPP */
