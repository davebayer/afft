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

#ifndef AFFT_DETAIL_CUDA_ENVIROMENT_HPP
#define AFFT_DETAIL_CUDA_ENVIROMENT_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

// Check if the CUDA Toolkit root directory is defined
#ifndef AFFT_CUDA_ROOT_DIR
# error "CUDA root directory is not defined"
#endif

namespace afft::detail::cuda
{
  /**
   * @brief Get the root directory of the CUDA Toolkit.
   * @return The root directory of the CUDA Toolkit.
   */
  [[nodiscard]] inline constexpr std::string_view getRootDir()
  {
    return {AFFT_CUDA_ROOT_DIR};
  }

  /**
   * @brief Get the path to the CUDA Toolkit include directory.
   * @return The path to the CUDA Toolkit include directory.
   */
  [[nodiscard]] inline constexpr std::string_view getIncludePath()
  {
    return {AFFT_CUDA_ROOT_DIR "/include"};
  }
} // namespace afft::detail::cuda

#endif /* AFFT_DETAIL_CUDA_ENVIROMENT_HPP */
