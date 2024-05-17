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
#include "detail/utils.hpp"

AFFT_EXPORT namespace afft
{
  /**
   * @brief Get the alignment of the pointers
   * @param ptrs Pointers
   * @return Alignment
   */
  template<typename... Args>
  [[nodiscard]] constexpr Alignment getAlignment(const Args*... ptrs) noexcept
  {
    static_assert(sizeof...(Args) > 0, "At least one pointer must be provided");

    auto getPtrAlignment = [](const std::uintptr_t uintPtr) constexpr -> Alignment
    {
      return static_cast<Alignment>(uintPtr & ~(uintPtr - 1));
    };

    return std::min({getPtrAlignment(reinterpret_cast<std::uintptr_t>(ptrs))...});
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

    return PrecisionTriad{/* .execution   = */ TypeProperties<std::remove_cv_t<PrecT>>::precision,
                          /* .source      = */ TypeProperties<std::remove_cv_t<PrecT>>::precision,
                          /* .destination = */ TypeProperties<std::remove_cv_t<PrecT>>::precision};
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

    return PrecisionTriad{/* .execution   = */ TypeProperties<std::remove_cv_t<ExecT>>::precision,
                          /* .source      = */ TypeProperties<std::remove_cv_t<MemoryT>>::precision,
                          /* .destination = */ TypeProperties<std::remove_cv_t<MemoryT>>::precision};
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

    return PrecisionTriad{/* .execution   = */ TypeProperties<std::remove_cv_t<ExecT>>::precision,
                          /* .source      = */ TypeProperties<std::remove_cv_t<SrcT>>::precision,
                          /* .destination = */ TypeProperties<std::remove_cv_t<DstT>>::precision};
  }

  /**
   * @brief Make strides.
   * @tparam extent Extent of the shape
   * @param shape Shape
   * @param fastestAxisStride Stride of the fastest axis
   * @return Strides
   */
  template<std::size_t extent>
  [[nodiscard]] constexpr auto makeStrides(View<std::size_t, extent> shape, std::size_t fastestAxisStride = 1)
    -> AFFT_RET_REQUIRES(AFFT_DETAIL_EXPAND(std::array<std::size_t, extent>), extent != dynamicExtent)
  {
    if (detail::cxx::any_of(shape.begin(), shape.end(), detail::IsZero<>{}))
    {
      throw std::invalid_argument("Shape must not contain zeros");
    }

    std::array<std::size_t, extent> strides{};

    if (!shape.empty())
    {
      strides[shape.size() - 1] = fastestAxisStride;

      for (std::size_t i = shape.size() - 1; i > 0; --i)
      {
        strides[i - 1] = shape[i] * strides[i];
      }
    }

    return strides;
  }

  /**
   * @brief Make strides.
   * @tparam extent Extent of the shape
   * @param shape Shape
   * @param fastestAxisStride Stride of the fastest axis
   * @return Strides
   */
  template<std::size_t extent>
  [[nodiscard]] auto makeStrides(View<std::size_t, extent> shape, std::size_t fastestAxisStride = 1)
    -> AFFT_RET_REQUIRES(AFFT_DETAIL_EXPAND(std::vector<std::size_t>), extent == dynamicExtent)
  {
    if (detail::cxx::any_of(shape.begin(), shape.end(), detail::IsZero<>{}))
    {
      throw std::invalid_argument("Shape must not contain zeros");
    }

    std::vector<std::size_t> strides(resultShape.size());

    if (!shape.empty())
    {
      strides[shape.size() - 1] = fastestAxisStride;

      for (std::size_t i = shape.size() - 1; i > 0; --i)
      {
        strides[i - 1] = shape[i] * strides[i];
      }
    }

    return strides;
  }

  /**
   * @brief Make transposed strides.
   * @tparam extent Extent of the shape
   * @param resultShape Shape of the result
   * @param orgAxesOrder Original axes order
   * @param fastestAxisStride Stride of the fastest axis
   * @return Strides
   */
  template<std::size_t extent = dynamicExtent>
  [[nodiscard]] auto makeTransposedStrides(View<std::size_t, extent> resultShape,
                                           View<std::size_t>         orgAxesOrder,
                                           std::size_t               fastestAxisStride = 1)
  {
    using ReturnT = std::conditional_t<(extent == dynamicExtent),
                                       std::array<std::size_t, extent>,
                                       std::vector<std::size_t>>;

    // If the axes order is empty, then the result axes order is the same as the original axes order
    if (orgAxesOrder.empty())
    {
      return makeStrides(resultShape, fastestAxisStride);
    }
    // Check if the axes size matches the shape size
    else if (orgAxesOrder.size() != shape.size())
    {
      throw std::invalid_argument("Axes order must have the same size as the shape");
    }

    // Check if the shape contains zeros
    if (detail::cxx::any_of(resultShape.begin(), resultShape.end(), detail::IsZero<>{}))
    {
      throw std::invalid_argument("Shape must not contain zeros");
    }
    
    // Check if the axes order contains out-of-range values or duplicates
    for (std::size_t i{}; i < orgAxesOrder.size(); ++i)
    {
      if (orgAxesOrder[i] >= orgAxesOrder.size())
      {
        throw std::invalid_argument("Axes order must not contain out-of-range values");
      }

      for (std::size_t j{i + 1}; j < axes.size(); ++j)
      {
        if (orgAxesOrder[i] == orgAxesOrder[j])
        {
          throw std::invalid_argument("Axes order must not contain duplicates");
        }
      }
    }

    ReturnT strides{};

    // Resize the strides if the extent is dynamic
    if constexpr (extent == dynamicExtent)
    {
      strides.resize(resultShape.size());
    }

    // Calculate the strides
    if (const std::size_t size = resultShape.size(); size > 0)
    {
      strides[orgAxesOrder[size - 1]] = fastestAxisStride;

      for (std::size_t i = size - 1; i > 0; --i)
      {
        strides[orgAxesOrder[i - 1]] = resultShape[i] * strides[orgAxesOrder[i]];
      }
    }

    return strides;
  }
} // namespace afft

#endif /* AFFT_UTILS_HPP */
