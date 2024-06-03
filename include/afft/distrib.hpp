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

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "common.hpp"

AFFT_EXPORT namespace afft
{
  /**
   * @struct MemoryBlock
   * @tparam rank Rank of the memory block, dynamic by default
   * @brief Memory block
   */
  template<std::size_t rank = dynamicRank>
  struct MemoryBlock
  {
    View<std::size_t, rank> starts{};  ///< starts of the memory block
    View<std::size_t, rank> sizes{};   ///< sizes of the memory block
    View<std::size_t, rank> strides{}; ///< strides of the memory block
  };

inline namespace spst
{
  /**
   * @struct MemoryLayout
   * @tparam rank Rank of the memory layout, dynamic by default
   * @brief Memory layout
   */
  template<std::size_t rank = dynamicRank>
  struct MemoryLayout
  {
    View<std::size_t, rank> srcStrides{}; ///< stride of the source data
    View<std::size_t, rank> dstStrides{}; ///< stride of the destination data
  };
} // inline namespace spst

namespace spmt
{
  /**
   * @struct MemoryLayout
   * @tparam rank Rank of the memory layout, dynamic by default
   * @brief Memory layout
   */
  template<std::size_t rank = dynamicRank>
  struct MemoryLayout
  {
    View<MemoryBlock<rank>> srcBlocks{};    ///< source memory blocks
    View<MemoryBlock<rank>> dstBlocks{};    ///< destination memory blocks
    View<std::size_t, rank> srcAxesOrder{}; ///< order of the source axes
    View<std::size_t, rank> dstAxesOrder{}; ///< order of the destination axes
  };
} // namespace spmt

namespace mpst
{
  /**
   * @struct MemoryLayout
   * @tparam rank Rank of the memory layout, dynamic by default
   * @brief Memory layout
   */
  template<std::size_t rank = dynamicRank>
  struct MemoryLayout
  {
    MemoryBlock<rank>       srcBlock{};     ///< source memory block
    MemoryBlock<rank>       dstBlock{};     ///< destination memory block
    View<std::size_t, rank> srcAxesOrder{}; ///< order of the source axes
    View<std::size_t, rank> dstAxesOrder{}; ///< order of the destination axes
  };
} // namespace mpst

  /// @brief Alias for single process, single target namespace
  namespace single = spst;

  /// @brief Alias for single process, multiple targets namespace
  namespace multi = spmt;

#if AFFT_MP_BACKEND_IS(MPI)
  /// @brief Alias for multiple processes, single target namespace
  namespace mpi = mpst;
#endif
} // namespace afft

#endif /* AFFT_DISTRIB_HPP */
