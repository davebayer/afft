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

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "common.hpp"
#include "concepts.hpp"

namespace afft
{
  /**
   * @brief Get the alignment of the pointers
   * @param ptrs Pointers
   * @return Alignment
   */
  [[nodiscard]] constexpr Alignment getAlignment(const auto*... ptrs)
    requires (sizeof...(ptrs) > 0)
  {
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
  template<KnownType PrecT>
  [[nodiscard]] constexpr PrecisionTriad makePrecision() noexcept
  {
    return PrecisionTriad{/* .execution   = */ TypeProperties<std::remove_cvref_t<PrecT>>::precision,
                          /* .source      = */ TypeProperties<std::remove_cvref_t<PrecT>>::precision,
                          /* .destination = */ TypeProperties<std::remove_cvref_t<PrecT>>::precision};
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
    return PrecisionTriad{/* .execution   = */ TypeProperties<std::remove_cvref_t<ExecT>>::precision,
                          /* .source      = */ TypeProperties<std::remove_cvref_t<MemoryT>>::precision,
                          /* .destination = */ TypeProperties<std::remove_cvref_t<MemoryT>>::precision};
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
    return PrecisionTriad{/* .execution   = */ TypeProperties<std::remove_cvref_t<ExecT>>::precision,
                          /* .source      = */ TypeProperties<std::remove_cvref_t<SrcT>>::precision,
                          /* .destination = */ TypeProperties<std::remove_cvref_t<DstT>>::precision};
  }
} // namespace afft

#endif /* AFFT_UTILS_HPP */
