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

#ifndef CONVERT_HPP
#define CONVERT_HPP

#include <afft/afft.h>
#include <afft/afft.hpp>

/**
 * @brief Compare two enum values.
 * @param lhs Left-hand side value.
 * @param rhs Right-hand side value.
 * @return True if the values are equal, false otherwise.
 * @note This function is only available for enum types.
 */
template<typename T, typename U>
[[nodiscard]] constexpr auto operator==(const T& lhs, const U& rhs) noexcept
  -> AFFT_RET_REQUIRES(bool, std::is_enum_v<T> && std::is_enum_v<U>)
{
  std::underlying_type_t<T> lhsValue = static_cast<std::underlying_type_t<T>>(lhs);
  std::underlying_type_t<U> rhsValue = static_cast<std::underlying_type_t<U>>(rhs);

  return lhsValue == rhsValue;
}

/**
 * @brief Enum conversion base class.
 * @tparam CxxE C++ enum type.
 * @tparam CE C enum type.
 * @tparam cToCxxCvtError Error to throw when converting from C to C++.
 */
template<typename CxxE, typename CE, afft_Error cToCxxCvtError = afft_Error_success>
struct EnumConvertBase
{
  static_assert(std::is_enum_v<CxxE>, "E must be an enum type");
  static_assert(std::is_same_v<CE, std::underlying_type_t<CxxE>>, "CE must be the underlying type of E");
  
  using CxxType = CxxE; ///< C++ enum type.
  using CType   = CE;   ///< C enum type.

  static constexpr afft_Error error = cToCxxCvtError; ///< Error to throw when converting from C to C++.

  /**
   * @brief Convert from C to C++.
   * @param cValue C enum value.
   * @return C++ enum value.
   */
  [[nodiscard]] static constexpr CxxType fromC(CType cValue) noexcept(error == afft_Error_success)
  {
    const auto cxxValue = static_cast<CxxType>(cValue);

    if constexpr (error != afft_Error_success)
    {
      if (!afft::detail::isValid(cxxValue))
      {
        throw error;
      }
    }

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ enum value.
   * @return C enum value.
   */
  [[nodiscard]] static constexpr CType toC(CxxType cxxValue)
  {
    if (!afft::detail::isValid(cxxValue))
    {
      throw afft_Error_internal;
    }

    return static_cast<CType>(afft::detail::cxx::to_underlying(cxxValue));
  }
};

/**
 * @brief Struct conversion base class.
 * @tparam CxxS C++ struct type.
 * @tparam CS C struct type.
 */
template<typename CxxS, typename CS>
struct StructConvertBase
{
  using CxxType = CxxS; ///< C++ struct type.
  using CType   = CS;   ///< C struct type.
};

/**
 * @brief Convert between C and C++ types.
 * @tparam CxxT C++ type.
 */
template<typename CxxT>
struct Convert;

#endif /* CONVERT_HPP */
