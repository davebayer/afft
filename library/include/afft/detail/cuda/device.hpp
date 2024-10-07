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

#ifndef AFFT_DETAIL_CUDA_DEVICE_HPP
#define AFFT_DETAIL_CUDA_DEVICE_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "error.hpp"

namespace afft::detail::cuda
{
  /**
   * @brief Get the number of CUDA devices.
   * @return The number of CUDA devices.
   */
  [[nodiscard]] inline int getDeviceCount()
  {
    int count{};
    checkError(cudaGetDeviceCount(&count));
    return count;
  }

  /**
   * @brief Check if a device is valid.
   * @param device The device to check.
   * @return True if the device is valid, false otherwise.
   */
  [[nodiscard]] inline bool isValidDevice(int device)
  {
    return (device >= 0) && (device < getDeviceCount());
  }

  /**
   * @brief Get the current device.
   * @return The current device.
   */
  [[nodiscard]] inline int getCurrentDevice()
  {
    int device{};
    checkError(cudaGetDevice(&device));
    return device;
  }

  /// @brief Compute capability.
  struct ComputeCapability
  {
    int major{}; ///< Major version.
    int minor{}; ///< Minor version.

    /**
     * @brief Compare two compute capabilities.
     * @param lhs Left-hand side.
     * @param rhs Right-hand side.
     * @return True if the compute capabilities are equal, false otherwise.
     */
    [[nodiscard]] friend bool operator==(const ComputeCapability& lhs, const ComputeCapability& rhs) noexcept
    {
      return lhs.major == rhs.major && lhs.minor == rhs.minor;
    }

    /**
     * @brief Compare two compute capabilities.
     * @param lhs Left-hand side.
     * @param rhs Right-hand side.
     * @return True if the compute capabilities are not equal, false otherwise.
     */
    [[nodiscard]] friend bool operator!=(const ComputeCapability& lhs, const ComputeCapability& rhs) noexcept
    {
      return !(lhs == rhs);
    }

    /**
     * @brief Compare two compute capabilities.
     * @param lhs Left-hand side.
     * @param rhs Right-hand side.
     * @return True if the left-hand side is less than the right-hand side, false otherwise.
     */
    [[nodiscard]] friend bool operator<(const ComputeCapability& lhs, const ComputeCapability& rhs) noexcept
    {
      return (lhs.major < rhs.major) || (lhs.major == rhs.major && lhs.minor < rhs.minor);
    }

    /**
     * @brief Compare two compute capabilities.
     * @param lhs Left-hand side.
     * @param rhs Right-hand side.
     * @return True if the left-hand side is less than or equal to the right-hand side, false otherwise.
     */
    [[nodiscard]] friend bool operator<=(const ComputeCapability& lhs, const ComputeCapability& rhs) noexcept
    {
      return (lhs < rhs) || (lhs == rhs);
    }

    /**
     * @brief Compare two compute capabilities.
     * @param lhs Left-hand side.
     * @param rhs Right-hand side.
     * @return True if the left-hand side is greater than the right-hand side, false otherwise.
     */
    [[nodiscard]] friend bool operator>(const ComputeCapability& lhs, const ComputeCapability& rhs) noexcept
    {
      return !(lhs <= rhs);
    }

    /**
     * @brief Compare two compute capabilities.
     * @param lhs Left-hand side.
     * @param rhs Right-hand side.
     * @return True if the left-hand side is greater than or equal to the right-hand side, false otherwise.
     */
    [[nodiscard]] friend bool operator>=(const ComputeCapability& lhs, const ComputeCapability& rhs) noexcept
    {
      return !(lhs < rhs);
    }
  };

  /**
   * @brief Get the compute capability of a device.
   * @param device The device.
   * @return The compute capability.
   */
  [[nodiscard]] inline ComputeCapability getComputeCapability(int device)
  {
    if (!cuda::isValidDevice(device))
    {
      throw Exception{Error::invalidArgument, "invalid device"};
    }

    ComputeCapability cc{};

    checkError(cudaDeviceGetAttribute(&cc.major, cudaDevAttrComputeCapabilityMajor, device));
    checkError(cudaDeviceGetAttribute(&cc.minor, cudaDevAttrComputeCapabilityMinor, device));

    return cc;
  }

  /**
   * @brief Check if a device supports UVA.
   * @param device The device.
   * @return True if the device supports UVA, false otherwise.
   */
  [[nodiscard]] inline bool hasUva(int device)
  {
    if (!cuda::isValidDevice(device))
    {
      throw Exception{Error::invalidArgument, "invalid device"};
    }

    int hasUva{};
    checkError(cudaDeviceGetAttribute(&hasUva, cudaDevAttrUnifiedAddressing, device));
    return (hasUva != 0);
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
        checkError(cudaSetDevice(device));
      }

      /**
       * @brief Destroy the ScopedDevice object and restore the previous device.
       */
      ~ScopedDevice()
      {
        checkError(cudaSetDevice(mPrevDevice));
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
} // namespace afft::detail::cuda

#endif /* AFFT_DETAIL_CUDA_DEVICE_HPP */
