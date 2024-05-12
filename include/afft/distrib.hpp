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

#include <cstddef>

#include "common.hpp"
#include "mp.hpp"

namespace afft
{
namespace distrib
{
  /**
   * @enum Type
   * @brief Distribution type
   */
  enum class Type
  {
    spst,           ///< single process, single target
    spmt,           ///< single process, multiple targets
    mpst,           ///< multiple processes, single target
    
    single = spst,  ///< alias for single process, single target
    multi  = spmt,  ///< alias for single process, multiple targets
    mpi    = mpst,  ///< alias for multiple processes, single target
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
} // namespace distrib

namespace spst
{
  /// @brief Memory layout
  struct MemoryLayout
  {
    View<std::size_t> srcStrides{}; ///< stride of the source data
    View<std::size_t> dstStrides{}; ///< stride of the destination data
  };
} // namespace spst

namespace spmt
{
  /// @brief Memory layout
  struct MemoryLayout
  {
    View<distrib::MemoryBlock> srcBlocks{};    ///< source memory blocks
    View<distrib::MemoryBlock> dstBlocks{};    ///< destination memory blocks
    View<std::size_t>          srcAxesOrder{}; ///< order of the source axes
    View<std::size_t>          dstAxesOrder{}; ///< order of the destination axes
  };
} // namespace spmt

namespace mpst
{
  /// @brief Memory layout
  struct MemoryLayout
  {
    distrib::MemoryBlock srcBlock{};     ///< source memory block
    distrib::MemoryBlock dstBlock{};     ///< destination memory block
    View<std::size_t>    srcAxesOrder{}; ///< order of the source axes
    View<std::size_t>    dstAxesOrder{}; ///< order of the destination axes
  };
} // namespace mpst

  /// @brief Introduce single process, single target memory layout to the namespace
  using spst::MemoryLayout;

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
