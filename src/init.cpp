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

#include "afft/afft.hpp"

/**
 * @brief Initialize the library.
 * @param errDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error afft_init(afft_ErrorDetails* errDetails)
try
{
  afft::init();

  return afft_Error_success;
}
catch (...)
{
  return handleException(errDetails);
}

/**
 * @brief Finalize the library.
 * @param errDetails Error details.
 * @return Error code.
 */
extern "C" afft_Error afft_finalize(afft_ErrorDetails* errDetails)
try
{
  afft::finalize();
  
  return afft_Error_success;
}
catch (...)
{
  return handleException(errDetails);
}
