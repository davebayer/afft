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

#ifndef AFFT_DETAIL_MACRO_HPP
#define AFFT_DETAIL_MACRO_HPP

/**
 * @brief Expand a variadic argument list 256 times.
 * @param ... Variadic arguments.
 * @return Expanded arguments.
 */
#define AFFT_DETAIL_EXPAND(...) \
  AFFT_DETAIL_EXPAND4(AFFT_DETAIL_EXPAND4(AFFT_DETAIL_EXPAND4(AFFT_DETAIL_EXPAND4(__VA_ARGS__))))
#define AFFT_DETAIL_EXPAND4(...) \
  AFFT_DETAIL_EXPAND3(AFFT_DETAIL_EXPAND3(AFFT_DETAIL_EXPAND3(AFFT_DETAIL_EXPAND3(__VA_ARGS__))))
#define AFFT_DETAIL_EXPAND3(...) \
  AFFT_DETAIL_EXPAND2(AFFT_DETAIL_EXPAND2(AFFT_DETAIL_EXPAND2(AFFT_DETAIL_EXPAND2(__VA_ARGS__))))
#define AFFT_DETAIL_EXPAND2(...) \
  AFFT_DETAIL_EXPAND1(AFFT_DETAIL_EXPAND1(AFFT_DETAIL_EXPAND1(AFFT_DETAIL_EXPAND1(__VA_ARGS__))))
#define AFFT_DETAIL_EXPAND1(...) \
  __VA_ARGS__

/**
 * @brief Expand and concatenate two tokens.
 * @param x First token.
 * @param y Second token.
 * @return Concatenated tokens.
 */
#define AFFT_DETAIL_EXPAND_AND_CONCAT(x, y) \
  AFFT_DETAIL_EXPAND_AND_CONCAT_HELPER(x, y)
#define AFFT_DETAIL_EXPAND_AND_CONCAT_HELPER(x, y) \
  x##y

#endif /* AFFT_DETAIL_MACRO_HPP */
