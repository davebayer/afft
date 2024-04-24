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

#ifndef AFFT_DETAIL_GPU_CUDA_DEVICE_HPP
#define AFFT_DETAIL_GPU_CUDA_DEVICE_HPP

#include <cuda.h>
#include <cuda_runtime.h>

#include "error.hpp"

namespace afft::detail::gpu::cuda
{
  /**
   * @brief Get the number of CUDA devices.
   * @return The number of CUDA devices.
   */
  int getDeviceCount()
  {
    int count{};
    Error::check(cudaGetDeviceCount(&count));
    return count;
  }

  /**
   * @brief Check if a device is valid.
   * @param device The device to check.
   * @return True if the device is valid, false otherwise.
   */
  bool isValidDevice(int device)
  {
    return (device >= 0) && (device < getDeviceCount());
  }

  /**
   * @brief Get the current device.
   * @return The current device.
   */
  int getCurrentDevice()
  {
    int device{};
    Error::check(cudaGetDevice(&device));
    return device;
  }

  /**
   * @brief Scoped device sets the current device to the specified device and
   *        restores the previous device when the object goes out of scope.
   */
  class ScopedDevice
  {
    public:
      /**
       * @brief Construct a new ScopedDevice object.
       * @param device The device to set.
       */
      explicit ScopedDevice(int device)
      : mPrevDevice{getCurrentDevice()}
      {
        Error::check(cudaSetDevice(device));
      }

      /**
       * @brief Destroy the ScopedDevice object and restore the previous device.
       */
      ~ScopedDevice()
      {
        Error::check(cudaSetDevice(mPrevDevice));
      }

      /**
       * @brief Get the current device.
       * @return The current device.
       */
      [[nodiscard]] operator int() const
      {
        return getCurrentDevice();
      }
    protected:
    private:
      int mPrevDevice{}; ///< The previous device.
  };
} // namespace afft::detail::gpu::cuda

#endif /* AFFT_DETAIL_GPU_CUDA_DEVICE_HPP */
