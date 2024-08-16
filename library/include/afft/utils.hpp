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

#ifndef AFFT_UTILS_HPP
#define AFFT_UTILS_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "common.hpp"
#include "memory.hpp"
#include "detail/utils.hpp"

AFFT_EXPORT namespace afft
{
  /// @brief Free deleter for std::unique_ptr
  struct FreeDeleter
  {
    /**
     * @brief Free the pointer
     * @tparam T Type
     * @param ptr Pointer
     */
    template<typename T>
    void operator()(T* ptr) const noexcept
    {
      std::free(ptr);
    }
  };

  /**
   * @brief Make a scalar view
   * @tparam T Type
   * @param scalar Scalar
   * @return Scalar view
   */
  template<typename T>
  View<T> makeScalarView(const T& scalar) noexcept
  {
    return View<T>{&scalar, 1};
  }

  /**
   * @brief Get the alignment of the pointers
   * @tparam PtrTs Pointer types
   * @param ptrs Pointers
   * @return Alignment
   */
  template<typename... PtrTs>
  [[nodiscard]] Alignment alignmentOf(const PtrTs*... ptrs) noexcept
  {
    static_assert(sizeof...(ptrs) > 0, "At least one pointer must be provided");

    const auto bitOredPtrs = (0 | ... | reinterpret_cast<std::uintptr_t>(ptrs));

    return static_cast<Alignment>(bitOredPtrs & ~(bitOredPtrs - 1));
  }

  /**
   * @brief Make a precision triad
   * @tparam PrecT Precision type common for all three types, must be a known type
   * @return Precision triad
   */
  template<typename PrecT>
  [[nodiscard]] constexpr PrecisionTriad makePrecision() noexcept
  {
    static_assert(isKnownType<PrecT>, "Precision type must be a known type");

    return PrecisionTriad{/* .execution   = */ typePrecision<PrecT>,
                          /* .source      = */ typePrecision<PrecT>,
                          /* .destination = */ typePrecision<PrecT>};
  }

  /**
   * @brief Make a precision triad
   * @tparam ExecT Execution type, must be a known type
   * @tparam MemoryT Memory type, must be a known type, used for both source and destination
   * @return Precision triad
   */
  template<typename ExecT, typename MemoryT>
  [[nodiscard]] constexpr PrecisionTriad makePrecision() noexcept
  {
    static_assert(isKnownType<ExecT>, "Execution precision type must be a known type");
    static_assert(isKnownType<MemoryT>, "Memory precision type must be a known type");

    return PrecisionTriad{/* .execution   = */ typePrecision<ExecT>,
                          /* .source      = */ typePrecision<MemoryT>,
                          /* .destination = */ typePrecision<MemoryT>};
  }

  /**
   * @brief Make a precision triad
   * @tparam ExecT Execution type, must be a known type
   * @tparam SrcT Source type, must be a known type
   * @tparam DstT Destination type, must be a known type
   * @return Precision triad
   */
  template<typename ExecT, typename SrcT, typename DstT>
  [[nodiscard]] constexpr PrecisionTriad makePrecision() noexcept
  {
    static_assert(isKnownType<ExecT>, "Execution precision type must be a known type");
    static_assert(isKnownType<SrcT>, "Source precision type must be a known type");
    static_assert(isKnownType<DstT>, "Destination precision type must be a known type");

    return PrecisionTriad{/* .execution   = */ typePrecision<ExecT>,
                          /* .source      = */ typePrecision<SrcT>,
                          /* .destination = */ typePrecision<DstT>};
  }

  /**
   * @brief Make strides.
   * @param shapeRank Shape rank
   * @param shape Shape
   * @param fastestAxisStride Stride of the fastest axis
   * @param strides Strides
   */
  constexpr void makeStrides(const std::size_t shapeRank,
                             const Size*       shape,
                             Size*             strides,
                             const Size        fastestAxisStride = 1)
  {
    if (shapeRank == 0)
    {
      throw Exception{Error::invalidArgument, "shape rank must be greater than zero"};
    }

    if (shape == nullptr)
    {
      throw Exception{Error::invalidArgument, "invalid shape"};
    }

    if (strides == nullptr)
    {
      throw Exception{Error::invalidArgument, "invalid strides"};
    }

    if (fastestAxisStride == 0)
    {
      throw Exception{Error::invalidArgument, "fastest axis stride must be greater than zero"};
    }

    if (detail::cxx::any_of(shape, shape + shapeRank, detail::IsZero<Size>{}))
    {
      throw Exception{Error::invalidArgument, "shape must not contain zeros"};
    }

    strides[shapeRank - 1] = fastestAxisStride;

    for (std::size_t i = shapeRank - 1; i > 0; --i)
    {
      strides[i - 1] = shape[i] * strides[i];
    }
  }

  /**
   * @brief Make transposed strides.
   * @param shapeRank Shape rank
   * @param shape Shape
   * @param orgAxesOrder Original axes order
   * @param strides Strides
   * @param fastestAxisStride Stride of the fastest axis
   */
  inline void makeTransposedStrides(const std::size_t shapeRank,
                                    const Size*       shape,
                                    const Axis*       orgAxesOrder,
                                    Size*             strides,
                                    const Size        fastestAxisStride = 1)
  {
    if (shapeRank == 0)
    {
      throw Exception{Error::invalidArgument, "shape rank must be greater than zero"};
    }

    if (shape == nullptr)
    {
      throw Exception{Error::invalidArgument, "invalid shape"};
    }

    if (orgAxesOrder == nullptr)
    {
      throw Exception{Error::invalidArgument, "invalid axes order"};
    }

    if (strides == nullptr)
    {
      throw Exception{Error::invalidArgument, "invalid strides"};
    }

    if (fastestAxisStride == 0)
    {
      throw Exception{Error::invalidArgument, "fastest axis stride must be greater than zero"};
    }

    if (detail::cxx::any_of(shape, shape + shapeRank, detail::IsZero<Size>{}))
    {
      throw Exception{Error::invalidArgument, "shape must not contain zeros"};
    }

    // Check if axes order is valid
    {
      std::bitset<maxDimCount> seenAxes{};

      for (std::size_t i{}; i < shapeRank; ++i)
      {
        if (orgAxesOrder[i] >= shapeRank)
        {
          throw Exception{Error::invalidArgument, "axes order must not contain out-of-range values"};
        }

        if (seenAxes.test(orgAxesOrder[i]))
        {
          throw Exception{Error::invalidArgument, "axes order must not contain duplicates"};
        }

        seenAxes.set(orgAxesOrder[i]);
      }
    }

    strides[orgAxesOrder[shapeRank - 1]] = fastestAxisStride;

    for (std::size_t i = shapeRank - 1; i > 0; --i)
    {
      strides[orgAxesOrder[i - 1]] = shape[i] * strides[orgAxesOrder[i]];
    }
  }
} // namespace afft

#endif /* AFFT_UTILS_HPP */
