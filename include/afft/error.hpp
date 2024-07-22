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

#ifndef AFFT_ERROR_HPP
#define AFFT_ERROR_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

AFFT_EXPORT namespace afft
{
  /// @brief Maximum error message size
  inline constexpr std::size_t maxErrorMessageSize{128};

  /// @brief Error codes
  enum class Error
  {
    // General errors
    internal = 1,    ///< Internal error
    invalidArgument, ///< Invalid argument
    invalidPlan,     ///< Invalid plan
    // Multi-process errors
    mpi,             ///< MPI error
    // Target errors
    cudaDriver,      ///< CUDA driver error
    cudaRuntime,     ///< CUDA runtime error
    cudaRtc,         ///< CUDA RTC error
    hip,             ///< HIP error
    opencl,          ///< OpenCL error
    // Backend errors
    clfft,           ///< clFFT error
    cufft,           ///< cuFFT error
    fftw3,           ///< FFTW3 error
    heffte,          ///< HeFFTe error
    hipfft,          ///< hipFFT error
    mkl,             ///< MKL error
    pocketfft,       ///< PocketFFT error
    rocfft,          ///< rocFFT error
    vkfft,           ///< VkFFT error
  };

  /// @brief Error return value returned by other libraries
  using ErrorRetval = std::variant<std::monostate
# ifdef AFFT_ENABLE_MPI
                                 , int
# endif
# ifdef AFFT_ENABLE_CUDA
                                 , CUresult
                                 , cudaError_t
                                 , nvrtcResult
# endif
# ifdef AFFT_ENABLE_HIP
                                 , hipError_t
# endif
# ifdef AFFT_ENABLE_OPENCL
                                 , cl_int
# endif
                                   >;


  /// @brief Exception class
  class Exception : public std::exception
  {
    public:
      /// @brief Default constructor
      Exception() noexcept = default;

      /**
       * @brief Constructor
       * @tparam RetvalT Error return value type
       * @param error Error code
       * @param what Error message
       * @param retval Error return value
       */
      template<typename RetvalT = std::monostate>
      explicit Exception(Error error, const char* what, RetvalT&& retval = {}) noexcept
      : mError{error}, mErrorRetval{std::forward<RetvalT>(retval)}
      {
        std::strncpy(mMessage, what, maxErrorMessageSize);
        mMessage[maxErrorMessageSize - 1] = '\0';
      }

      /// @brief Copy constructor (default)
      constexpr Exception(const Exception&) noexcept = default;

      /// @brief Move constructor (default)
      constexpr Exception(Exception&&) noexcept = default;

      /// @brief Destructor (default)
      virtual ~Exception() noexcept = default;

      /// @brief Copy assignment operator (default)
      Exception& operator=(const Exception&) noexcept = default;

      /// @brief Move assignment operator (default)
      Exception& operator=(Exception&&) noexcept = default;

      /**
       * @brief Get error code
       * @return Error code
       */
      [[nodiscard]] constexpr Error getError() const noexcept
      {
        return mError;
      }

      /**
       * @brief Get error message
       * @return Error message
       */
      [[nodiscard]] virtual const char* what() const noexcept override
      {
        return mMessage;
      }

      /**
       * @brief Get error return value
       * @return Error return value
       */
      [[nodiscard]] constexpr const ErrorRetval& getErrorRetval() const noexcept
      {
        return mErrorRetval;
      }
    private:
      Error       mError{};                        ///< Error code
      ErrorRetval mErrorRetval{};                  ///< Error return value
      char        mMessage[maxErrorMessageSize]{}; ///< Error message
  };
} // namespace afft

#endif /* AFFT_ERROR_HPP */
