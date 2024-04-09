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

#ifndef AFFT_DETAIL_GPU_CUFFT_HANDLE_HPP
#define AFFT_DETAIL_GPU_CUFFT_HANDLE_HPP

#include <cufft.h>

#include "error.hpp"

namespace afft::detail::gpu::cufft
{
  /**
   * @class Handle
   * @brief RAII wrapper for cuFFT handle.
   */
  class Handle
  {
    public:
      /// @brief Default constructor.
      Handle()
      {
        Error::check(cufftCreate(&mHandle));
      }

      /// @brief Deleted copy constructor.
      Handle(const Handle&) = delete;

      /// @brief Default move constructor.
      Handle(Handle&&) = default;

      /// @brief Deleted copy assignment operator.
      Handle& operator=(const Handle&) = delete;

      /// @brief Default move assignment operator.
      Handle& operator=(Handle&&) = default;

      /// @brief Destructor.
      ~Handle()
      {
        if (mHandle)
        {
          cufftDestroy(mHandle);
        }
      }

      /// @brief Get cuFFT handle.
      [[nodiscard]] operator cufftHandle() const
      {
        return mHandle;
      }
    private:
      cufftHandle mHandle{}; ///< cuFFT handle.
  };
} // namespace afft::detail::gpu::cufft

#endif /* AFFT_DETAIL_GPU_CUFFT_HANDLE_HPP */
