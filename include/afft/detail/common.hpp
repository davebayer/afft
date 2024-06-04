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

#ifndef AFFT_DETAIL_COMMON_HPP
#define AFFT_DETAIL_COMMON_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "utils.hpp"
#include "validate.hpp"
#include "../common.hpp"
#include "../cpu.hpp"
#include "../distrib.hpp"
#include "../gpu.hpp"

namespace afft::detail
{
  /**
   * @brief MaxDimArray is a std::array with a maximum number of elements defined by maxDimCount.
   * @tparam T The type of the elements.
   */
  template<typename T>
  struct MaxDimArray : std::array<T, maxDimCount>
  {
    /**
     * @brief Safely casts the elements of the array to a different type.
     * @tparam U The type to cast to.
     * @return A new MaxDimArray with the elements cast to the new type.
     */
    template<typename U>
    [[nodiscard]] constexpr MaxDimArray<U> cast() const noexcept(std::is_same_v<T, U>)
    {
      static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "Both types must be integral.");

      if constexpr (std::is_same_v<T, U>)
      {
        return *this;
      }
      else
      {
        MaxDimArray<U> result{};

        for (std::size_t i{}; i < this->size(); ++i)
        {
          result[i] = safeIntCast<U>((*this)[i]);
        }

        return result;
      }
    }
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_COMMON_HPP */
