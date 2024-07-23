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

#ifndef AFFT_INIT_HPP
#define AFFT_INIT_HPP

#include "detail/init.hpp"

AFFT_EXPORT namespace afft
{
  /// @brief Initialize the library. Should be called before any other afft function.
  void init();

  /// @brief Finalize the library.
  void finalize();

  namespace c
  {
    /**
     * @brief Initialize the library. Should be called before any other afft function.
     * @param errorDetails Error details.
     * @return Error code.
     */
    [[nodiscard]] inline Error init(ErrorDetails* errorDetails = nullptr) noexcept
    {
      return ::afft_init(errorDetails);
    }

    /**
     * @brief Finalize the library.
     * @param errorDetails Error details.
     * @return Error code.
     */
    [[nodiscard]] inline Error finalize(ErrorDetails* errorDetails = nullptr) noexcept
    {
      return ::afft_finalize(errorDetails);
    }
  } // namespace c

#ifdef AFFT_HEADER_ONLY
  AFFT_HEADER_ONLY_INLINE void init()
  {
    detail::Initializer::getInstance().init();
  }

  /// @brief Finalize the library.
  AFFT_HEADER_ONLY_INLINE void finalize()
  {
    detail::Initializer::getInstance().finalize();
  }
#endif
} // namespace afft

#endif /* AFFT_INIT_HPP */
