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

#ifndef HELPERS_CUDA_H
#define HELPERS_CUDA_H

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Check CUDA runtime error and exit if not success. Should not be used directly, use CUDART_CALL macro instead.
 * @param[in] error CUDA runtime error
 * @param[in] file  file name
 * @param[in] line  line number
 */
static inline void check_cudart_error(cudaError_t error, const char* file, int line)
{
  if (error != cudaSuccess)
  {
    fprintf(stderr, "CUDA error (%s:%d) - %s\n", file, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Macro for checking CUDA runtime errors. The call cannot contain _error variable.
 * @param[in] call CUDA runtime function call
 */
#define CUDART_CALL(call) check_cudart_error((call), __FILE__, __LINE__)

/**
 * @brief Get the number of CUDA devices
 * @return Number of CUDA devices
 */
static inline int helpers_cuda_getDeviceCount()
{
  int deviceCount;

  CUDART_CALL(cudaGetDeviceCount(&deviceCount));

  return deviceCount;
}

#ifdef __cplusplus
}
#endif

#endif /* HELPERS_CUDA_H */
