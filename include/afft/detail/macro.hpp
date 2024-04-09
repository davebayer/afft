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

/// @brief Expand variadic arguments 256 times.
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

/// @brief Macro expanding to parentheses. Useful for expanding variadic arguments.
#define AFFT_DETAIL_PARENS  ()

/// @brief Empty delimiter macro implementation.
#define AFFT_DETAIL_DELIM_EMPTY()
/// @brief Comma delimiter macro implementation.
#define AFFT_DETAIL_DELIM_COMMA() ,

/// @brief For each macro with delimiter implementation.
#define AFFT_DETAIL_FOR_EACH_WITH_DELIM(macro, delimMacro, ...) \
  __VA_OPT__(AFFT_DETAIL_EXPAND(AFFT_DETAIL_FOR_EACH_WITH_DELIM_HELPER(macro, delimMacro, __VA_ARGS__)))
#define AFFT_DETAIL_FOR_EACH_WITH_DELIM_HELPER(macro, delimMacro, elem, ...) \
  macro(elem) \
  __VA_OPT__(delimMacro AFFT_DETAIL_PARENS \
             AFFT_DETAIL_FOR_EACH_WITH_DELIM_HELPER_AGAIN AFFT_DETAIL_PARENS (macro, delimMacro, __VA_ARGS__))
#define AFFT_DETAIL_FOR_EACH_WITH_DELIM_HELPER_AGAIN() AFFT_DETAIL_FOR_EACH_WITH_DELIM_HELPER

/// @brief Variadic bit-or macro implementation.
#define AFFT_DETAIL_BITOR(...) \
  (0 __VA_OPT__(| AFFT_DETAIL_EXPAND(AFFT_DETAIL_BITOR_HELPER(__VA_ARGS__))))
#define AFFT_DETAIL_BITOR_HELPER(elem, ...) \
  elem __VA_OPT__(| AFFT_DETAIL_BITOR_HELPER_AGAIN AFFT_DETAIL_PARENS (__VA_ARGS__))
#define AFFT_DETAIL_BITOR_HELPER_AGAIN() AFFT_DETAIL_BITOR_HELPER

#endif /* AFFT_DETAIL_MACRO_HPP */
