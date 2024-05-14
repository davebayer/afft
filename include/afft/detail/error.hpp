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

#ifndef AFFT_DETAIL_ERROR_HPP
#define AFFT_DETAIL_ERROR_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "utils.hpp"

namespace afft::detail
{
#if defined(AFFT_DEBUG) && defined(__cpp_lib_source_location)
  /**
   * @brief Creates an exception with source location information. Only available in debug mode when C++20
   *        source_location is supported.
   * @tparam E Exception type.
   * @param msg Message.
   * @param loc Source location.
   * @return Exception.
   */
  template<typename E>
  [[nodiscard]] E makeException(std::string_view msg, std::source_location loc = std::source_location::current())
  {
    static_assert(std::is_base_of_v<std::exception, E>, "E must be derived from std::exception");

    return E{cformat("%s (%s:" PRIuLEAST32 ")", msg.data(), loc.file_name(), loc.line())};
  }
#else
  /**
   * @brief Creates an exception.
   * @tparam E Exception type.
   * @param msg Message.
   * @return Exception.
   */
  template<typename E>
  [[nodiscard]] E makeException(std::string_view msg)
  {
    static_assert(std::is_base_of_v<std::exception, E>, "E must be derived from std::exception");

    return E{msg.data()};
  }
#endif

  /**
   * @struct Error
   * @brief Class for return value error checking.
   */
  struct Error
  {
# if defined(AFFT_DEBUG) && defined(__cpp_lib_source_location)
    /**
     * @brief Checks the return value and throws an exception if it is not OK. Only available in debug mode when C++20
     *        source_location is supported.
     * @tparam R Checked return type.
     * @param result Return value.
     * @param loc Source location.
     */
    template<typename R>
    static void check(R result, std::source_location loc = std::source_location::current())
    {
      if (!isOk(result))
      {
        throw makeException<std::runtime_error>(makeErrorMessage(result), loc);
      }
    }
# else
    /**
     * @brief Checks the return value and throws an exception if it is not OK.
     * @tparam R Checked return type.
     * @param result Return value.
     */
    template<typename R>
    static void check(R result)
    {
      if (!isOk(result))
      {
        throw makeException<std::runtime_error>(makeErrorMessage(result));
      }
    }
# endif

    /**
     * @brief Checks the return value. Should be implemented by each backend.
     * @tparam R Checked return type.
     * @param result Return value.
     * @return true if the return value is OK, false otherwise.
     */
    template<typename R>
    [[nodiscard]] static constexpr bool isOk(R result);

    /**
     * @brief Creates an error message. Should be implemented by each backend.
     * @tparam R Checked return type.
     * @param result Return value.
     * @return Error message.
     */
    template<typename R>
    [[nodiscard]] static std::string makeErrorMessage(R result);
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_ERROR_HPP */
