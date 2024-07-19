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
   * @tparam I Integral type
   * @tparam shapeExt Shape extent
   * @tparam stridesExt Strides extent
   * @param shape Shape
   * @param fastestAxisStride Stride of the fastest axis
   * @param strides Strides
   */
  template<typename I, std::size_t stridesExt, std::size_t shapeExt>
  constexpr void makeStrides(View<I, shapeExt>   shape,
                             std::size_t         fastestAxisStride,
                             Span<I, stridesExt> strides)
  {
    static_assert(std::is_integral_v<I>, "I must be an integral type");
    static_assert((stridesExt == dynamicExtent) ||
                  (shapeExt == dynamicExtent) ||
                  (stridesExt == shapeExt), "strides and shape must have the same size");

    if (strides.size() != shape.size())
    {
      throw std::invalid_argument("strides and shape must have the same size");
    }

    if (detail::cxx::any_of(shape.begin(), shape.end(), detail::IsZero<I>{}))
    {
      throw std::invalid_argument("shape must not contain zeros");
    }

    strides[shape.size() - 1] = fastestAxisStride;

    for (std::size_t i = shape.size() - 1; i > 0; --i)
    {
      strides[i - 1] = shape[i] * strides[i];
    }
  }

  /**
   * @brief Make strides with the fastest axis stride set to 1.
   * @tparam I Integral type
   * @tparam shapeExt Shape extent
   * @tparam stridesExt Strides extent
   * @param shape Shape
   * @param strides Strides
   */
  template<typename I, std::size_t stridesExt, std::size_t shapeExt>
  constexpr void makeStrides(View<I, shapeExt> shape, Span<I, stridesExt> strides)
  {
    makeStrides(shape, 1, strides);
  }

  /**
   * @brief Make strides.
   * @tparam I Integral type
   * @tparam shapeExt Shape extent
   * @param shape Shape
   * @param fastestAxisStride Stride of the fastest axis
   * @return Strides
   */
  template<typename I, std::size_t shapeExt>
  [[nodiscard]] constexpr auto makeStrides(View<I, shapeExt> shape, std::size_t fastestAxisStride = 1)
    -> AFFT_RET_REQUIRES(AFFT_PARAM(std::array<I, shapeExt>), shapeExt != dynamicExtent)
  {
    std::array<I, shapeExt> strides{};

    makeStrides(shape, fastestAxisStride, Span<I, shapeExt>{strides});

    return strides;
  }

  /**
   * @brief Make strides with the fastest axis stride set to 1.
   * @tparam I Integral type
   * @tparam shapeExt Shape extent
   * @param shape Shape
   * @return Strides
   */
  template<typename I, std::size_t shapeExt>
  [[nodiscard]] auto makeStrides(View<I, shapeExt> shape, std::size_t fastestAxisStride = 1)
    -> AFFT_RET_REQUIRES(AFFT_PARAM(std::vector<I>), shapeExt == dynamicExtent)
  {
    static_assert(std::is_integral_v<I>, "I must be an integral type");

    std::vector<I> strides(shape.size());

    makeStrides(shape, fastestAxisStride, strides);

    return strides;
  }

  /**
   * @brief Make transposed strides.
   * @tparam I Integral type
   * @tparam A Integral type
   * @tparam shapeExt Shape extent
   * @tparam axesExt Axes order extent
   * @tparam stridesExt Strides extent
   * @param shape Shape
   * @param orgAxesOrder Original axes order
   * @param fastestAxisStride Stride of the fastest axis
   * @param strides Strides
   */
  template<typename I, typename A, std::size_t stridesExt, std::size_t shapeExt, std::size_t axesExt>
  constexpr void makeTransposedStrides(View<I, shapeExt>   shape,
                                       View<A, axesExt>    orgAxesOrder,
                                       std::size_t         fastestAxisStride,
                                       Span<I, stridesExt> strides)
  {
    static_assert(std::is_integral_v<I>, "I must be an integral type");
    static_assert(std::is_integral_v<A>, "A must be an integral type");
    static_assert((stridesExt == dynamicExtent) ||
                  (shapeExt == dynamicExtent) ||
                  (stridesExt == shapeExt), "strides and shape must have the same size");
    static_assert((stridesExt == dynamicExtent) ||
                  (axesExt == dynamicExtent) ||
                  (stridesExt == axesExt), "strides and axes order must have the same size");

    if (orgAxesOrder.size() != shape.size())
    {
      throw std::invalid_argument("Axes order must have the same size as the shape");
    }

    if (detail::cxx::any_of(shape.begin(), shape.end(), detail::IsZero<I>{}))
    {
      throw std::invalid_argument("Shape must not contain zeros");
    }

    for (std::size_t i{}; i < orgAxesOrder.size(); ++i)
    {
      if (orgAxesOrder[i] >= orgAxesOrder.size())
      {
        throw std::invalid_argument("Axes order must not contain out-of-range values");
      }

      for (std::size_t j{i + 1}; j < orgAxesOrder.size(); ++j)
      {
        if (orgAxesOrder[i] == orgAxesOrder[j])
        {
          throw std::invalid_argument("Axes order must not contain duplicates");
        }
      }
    }

    strides[orgAxesOrder[shape.size() - 1]] = fastestAxisStride;

    for (std::size_t i = shape.size() - 1; i > 0; --i)
    {
      strides[orgAxesOrder[i - 1]] = shape[i] * strides[orgAxesOrder[i]];
    }
  }

  /**
   * @brief Make transposed strides with the fastest axis stride set to 1.
   * @tparam I Integral type
   * @tparam A Integral type
   * @tparam shapeExt Shape extent
   * @tparam axesExt Axes order extent
   * @tparam stridesExt Strides extent
   * @param shape Shape
   * @param orgAxesOrder Original axes order
   * @param strides Strides
   */
  template<typename I, typename A, std::size_t stridesExt, std::size_t shapeExt, std::size_t axesExt>
  constexpr void makeTransposedStrides(View<I, shapeExt>   shape,
                                       View<A, axesExt>    orgAxesOrder,
                                       Span<I, stridesExt> strides)
  {
    makeTransposedStrides(shape, orgAxesOrder, 1, strides);
  }

  /**
   * @brief Make transposed strides.
   * @tparam I Integral type
   * @tparam A Integral type
   * @tparam shapeExt Shape extent
   * @tparam axesExt Axes order extent
   * @param shape Shape
   * @param orgAxesOrder Original axes order
   * @param fastestAxisStride Stride of the fastest axis
   * @return Strides
   */  
  template<typename I, typename A, std::size_t shapeExt, std::size_t axesExt>
  constexpr auto makeTransposedStrides(View<I, shapeExt> shape,
                                       View<A, axesExt>  orgAxesOrder,
                                       std::size_t       fastestAxisStride = 1)
    -> AFFT_RET_REQUIRES(AFFT_PARAM(std::array<I, shapeExt>), shapeExt != dynamicExtent)
  {
    static_assert(std::is_integral_v<I>, "I must be an integral type");
    static_assert(std::is_integral_v<A>, "A must be an integral type");

    std::array<I, shapeExt> strides{};

    makeTransposedStrides(shape, orgAxesOrder, fastestAxisStride, Span<I, shapeExt>{strides});

    return strides;
  }

  /**
   * @brief Make transposed strides.
   * @tparam I Integral type
   * @tparam A Integral type
   * @tparam shapeExt Shape extent
   * @tparam axesExt Axes order extent
   * @param shape Shape
   * @param orgAxesOrder Original axes order
   * @return Strides
   */
  template<typename I, typename A, std::size_t shapeExt, std::size_t axesExt>
  auto makeTransposedStrides(View<I, shapeExt> shape,
                             View<A, axesExt>  orgAxesOrder,
                             std::size_t       fastestAxisStride = 1)
    -> AFFT_RET_REQUIRES(AFFT_PARAM(std::vector<I>), shapeExt == dynamicExtent)
  {
    static_assert(std::is_integral_v<I>, "I must be an integral type");
    static_assert(std::is_integral_v<A>, "A must be an integral type");

    std::vector<I> strides(shape.size());

    makeTransposedStrides(shape, orgAxesOrder, fastestAxisStride, strides);

    return strides;
  }
} // namespace afft

#endif /* AFFT_UTILS_HPP */
