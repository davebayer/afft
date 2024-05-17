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

#ifndef AFFT_DETAIL_GPU_CUDA_ERROR_HPP
#define AFFT_DETAIL_GPU_CUDA_ERROR_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../error.hpp"
#include "../utils.hpp"

namespace afft::detail
{
  /**
   * @brief Specialization of isOk method for cudaError_t.
   * @param error CUDA error.
   * @return True if error is cudaSuccess, false otherwise.
   */
  template<>
  [[nodiscard]] inline constexpr bool Error::isOk(cudaError_t error)
  {
    return (error == cudaSuccess);
  }

  /**
   * @brief Specialization of makeErrorMessage method for cudaError_t.
   * @param error CUDA error.
   * @return Error message.
   */
  template<>
  [[nodiscard]] inline std::string Error::makeErrorMessage(cudaError_t error)
  {
    return cformat("[CUDA Runtime error] %s - %s", cudaGetErrorName(error), cudaGetErrorString(error));
  }

  /**
   * @brief Specialization of isOk method for CUresult.
   * @param result CUDA driver error.
   * @return True if result is CUDA_SUCCESS, false otherwise.
   */
  template<>
  [[nodiscard]] inline constexpr bool Error::isOk(CUresult result)
  {
    return (result == CUDA_SUCCESS);
  }

  /**
   * @brief Specialization of makeErrorMessage method for CUresult.
   * @param result CUDA driver error.
   * @return Error message.
   */
  template<>
  [[nodiscard]] inline std::string Error::makeErrorMessage(CUresult result)
  {
    const char* errorName{};
    const char* errorStr{};

    cuGetErrorName(result, &errorName);
    cuGetErrorString(result, &errorStr);

    return cformat("[CUDA Driver error] %s - %s", (errorName != nullptr) ? errorName : "Unnamed error",
                                                  (errorStr != nullptr) ? errorName : "No description");
  }
} // namespace afft::detail

#endif /* AFFT_DETAIL_GPU_CUDA_ERROR_HPP */
