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

#ifndef AFFT_MACRO_HPP
#define AFFT_MACRO_HPP

#include "detail/macro.hpp"

/// @brief Macro for identity function
#define AFFT_IDENTITY(...) __VA_ARGS__

/// @brief Macro for emty delimiter
#define AFFT_DELIM_EMPTY  AFFT_DETAIL_DELIM_EMPTY
/// @brief Macro for comma delimiter
#define AFFT_DELIM_COMMA  AFFT_DETAIL_DELIM_COMMA

/**
 * @brief Macro for applying a macro to each variadic argument
 * @param macro Macro to apply
 * @param ... Variadic arguments
 * @return Macro applied to each variadic argument
 */
#define AFFT_FOR_EACH(macro, ...) \
  AFFT_FOR_EACH_WITH_DELIM(macro, AFFT_DELIM_EMPTY, __VA_ARGS__)

/**
 * @brief Macro for applying a macro to each variadic argument with a delimiter
 * @param macro Macro to apply
 * @param delimMacro Delimiter macro
 * @param ... Variadic arguments
 * @return Macro applied to each variadic argument with a delimiter
 */
#define AFFT_FOR_EACH_WITH_DELIM(macro, delimMacro, ...) \
  AFFT_DETAIL_FOR_EACH_WITH_DELIM(macro, delimMacro, __VA_ARGS__)

/// @brief Macro for bit-wise OR on variadic arguments
#define AFFT_BITOR(...) AFFT_DETAIL_BITOR(__VA_ARGS__)

#endif /* AFFT_MACRO_HPP */
