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

#ifndef HELPERS_AFFT_H
#define HELPERS_AFFT_H

#include <stdio.h>
#include <stdlib.h>

#include <afft/afft.h>

#ifdef __cplusplus
extern "C"
{
#endif

/// @brief Error details, must be defined in the main file
static afft_ErrorDetails errDetails;

/**
 * @brief Check afft error and exit if not success. Should not be used directly, use CALL_AFFT macro instead.
 * @param[in] error afft error
 * @param[in] file  file name
 * @param[in] line  line number
 */
static inline check_afft_error(afft_Error error, const char* file, int line)
{
  if (error != afft_Error_success)
  {
    fprintf(stderr, "afft error (%s:%d): %s\n", file, line, errDetails.message);
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Macro for checking afft errors. The call cannot contain _err variable.
 * @param[in] call afft function call
 */
#define CALL_AFFT(call) check_afft_error((call), __FILE__, __LINE__)

#ifdef __cplusplus
}
#endif

#endif /* HELPERS_AFFT_H */
