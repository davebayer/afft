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

#ifndef AFFT_DETAIL_HEFFTE_COMMON_HPP
#define AFFT_DETAIL_HEFFTE_COMMON_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../exception.hpp"
#include "../../Span.hpp"

namespace afft::detail::heffte
{
  /// @brief Alias for the index type.
  using Index = std::size_t;

  /// @brief Alias for the box type.
  using Box = ::heffte::box3d<Index>;

  /**
   * @brief Make a box from starts and sizes.
   * @param starts The starts.
   * @param sizes The sizes.
   * @return The box.
   */
  [[nodiscard]] inline Box makeBox(View<std::size_t> starts, View<std::size_t> sizes)
  {
    if (starts.size() > 3 || sizes.size() > 3)
    {
      throw std::invalid_argument("number of starts and sizes must be less than or equal to 3");
    }

    std::array<Index, 3> clow{};
    std::array<Index, 3> chigh{};

    std::copy(starts.begin(), starts.end(), clow.begin());
    std::transform(starts.begin(),
                   starts.end(),
                   sizes.begin(),
                   chigh.begin(),
                   [](const auto start, const auto size)
    {
      return start + size - 1;
    });

    return Box{clow, chigh};
  }

  /**
   * @brief Safe call a function.
   * @param fn The function to call.
   * @return The result of the function.
   */
  template<typename Fn>
  decltype(auto) safeCall(Fn&& fn)
  {
    static_assert(std::is_invocable_v<Fn>, "The function must be invocable.");

    try
    {
      return fn();
    }
    catch (...)
    {
      processException();
    }
  }

  /// @brief Process the current exception.
  [[noreturn]] inline void processException()
  {
    try
    {
      throw;
    }
    catch (const BackendError&)
    {
      throw;
    }
    catch (const std::exception& e)
    {
      throw BackendError{Backend::heffte, e.what()};
    }
    catch (...)
    {
      throw BackendError{Backend::heffte, "unknown error"};
    }
  }
} // namespace afft::detail::heffte

#endif /* AFFT_DETAIL_HEFFTE_COMMON_HPP */
