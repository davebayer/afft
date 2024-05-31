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

#ifndef AFFT_EXCEPTION_HPP
#define AFFT_EXCEPTION_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "backend.hpp"
#include "detail/utils.hpp"

namespace afft
{
  /// @brief Exception base class for afft library.
  struct Exception : public std::exception
  {
    /// @brief Default constructor
    Exception() noexcept = default;

    /// @brief Destructor
    virtual ~Exception() noexcept = default;
  };

  /**
   * @brief Exception thrown when a backend error occurs.
   */
  class BackendException : public Exception
  {
    public:
      /// @brief Default constructor (deleted)
      BackendException() = delete;

      /**
       * @brief Constructor.
       * @param backend Backend that caused the exception.
       * @param msg Message of the exception.
       */
      explicit BackendException(Backend backend, std::string_view msg)
      : mBackend{backend}, mMessage{makeMessage(backend, msg)}
      {}

      /// @brief Copy constructor
      BackendException(const BackendException&) = default;

      /// @brief Move constructor
      BackendException(BackendException&&) = default;

      /// @brief Destructor
      virtual ~BackendException() override = default;

      /// @brief Copy assignment operator
      BackendException& operator=(const BackendException&) = default;

      /// @brief Move assignment operator
      BackendException& operator=(BackendException&&) = default;

      /**
       * @brief Gets the message of the exception.
       * @return Message of the exception.
       */
      [[nodiscard]] virtual const char* what() const noexcept override
      {
        return mMessage.c_str();
      }

      /**
       * @brief Gets the backend that caused the exception.
       * @return Backend that caused the exception.
       */
      [[nodiscard]] Backend backend() const noexcept
      {
        return mBackend;
      }
    private:
      /**
       * @brief Makes the message of the exception.
       * @param backend Backend that caused the exception.
       * @param msg Message of the exception.
       * @return Message of the exception.
       */
      static std::string makeMessage(Backend backend, std::string_view msg) noexcept
      {
        return detail::cformatNothrow("[%s error] %s", toString(backend).data(), msg.data());
      }

      Backend     mBackend{}; ///< Backend that caused the exception
      std::string mMessage{}; ///< Message of the exception
  };

  class GpuBackendException : public Exception
  {
    public:
      /// @brief Default constructor (deleted)
      GpuBackendException() = delete;

      /**
       * @brief Constructor.
       * @param msg Message of the exception.
       */
      explicit GpuBackendException(std::string_view msg)
      : mMessage{makeMessage(msg)}
      {}

      /// @brief Copy constructor
      GpuBackendException(const GpuBackendException&) = default;

      /// @brief Move constructor
      GpuBackendException(GpuBackendException&&) = default;

      /// @brief Destructor
      virtual ~GpuBackendException() override = default;

      /// @brief Copy assignment operator
      GpuBackendException& operator=(const GpuBackendException&) = default;

      /// @brief Move assignment operator
      GpuBackendException& operator=(GpuBackendException&&) = default;

      /**
       * @brief Gets the message of the exception.
       * @return Message of the exception.
       */
      [[nodiscard]] virtual const char* what() const noexcept override
      {
        return mMessage.c_str();
      }
    private:
      /**
       * @brief Makes the message of the exception.
       * @param msg Message of the exception.
       * @return Message of the exception.
       */
      static std::string makeMessage(std::string_view msg) noexcept
      {
        std::string_view gpuBackendName = "<Invalid GPU backend>";

#     if AFFT_GPU_BACKEND_IS(CUDA)
        gpuBackendName = "CUDA";
#     elif AFFT_GPU_BACKEND_IS(HIP)
        gpuBackendName = "HIP";
#     elif AFFT_GPU_BACKEND_IS(OPENCL)
        gpuBackendName = "OpenCL";
#     endif

        return detail::cformatNothrow("[%s error] %s", gpuBackendName.data(), msg.data());
      }

      std::string mMessage{}; ///< Message of the exception  
  };

  class MpBackendException : public Exception
  {
    public:
      /// @brief Default constructor (deleted)
      MpBackendException() = delete;

      /**
       * @brief Constructor.
       * @param msg Message of the exception.
       */
      explicit MpBackendException(std::string_view msg)
      : mMessage{makeMessage(msg)}
      {}

      /// @brief Copy constructor
      MpBackendException(const MpBackendException&) = default;

      /// @brief Move constructor
      MpBackendException(MpBackendException&&) = default;

      /// @brief Destructor
      virtual ~MpBackendException() override = default;

      /// @brief Copy assignment operator
      MpBackendException& operator=(const MpBackendException&) = default;

      /// @brief Move assignment operator
      MpBackendException& operator=(MpBackendException&&) = default;

      /**
       * @brief Gets the message of the exception.
       * @return Message of the exception.
       */
      [[nodiscard]] virtual const char* what() const noexcept override
      {
        return mMessage.c_str();
      }
    private:
      /**
       * @brief Makes the message of the exception.
       * @param msg Message of the exception.
       * @return Message of the exception.
       */
      static std::string makeMessage(std::string_view msg) noexcept
      {
        std::string_view mpBackendName = "<Invalid multi-process backend>";

#     if AFFT_MP_BACKEND_IS(MPI)
        mpBackendName = "MPI";
#     endif

        return detail::cformatNothrow("[%s error] %s", mpBackendName.data(), msg.data());
      }

      std::string mMessage{}; ///< Message of the exception
  };
} // namespace afft

#endif /* AFFT_EXCEPTION_HPP */
