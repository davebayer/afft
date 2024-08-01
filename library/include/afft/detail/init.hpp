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

#ifndef AFFT_DETAIL_INIT_HPP
#define AFFT_DETAIL_INIT_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

namespace afft::detail
{
  class Initializer
  {
    public:
      /// @brief Clock type.
      using Clock     = std::chrono::steady_clock;

      /// @brief Time stamp type.
      using TimeStamp = std::chrono::time_point<Clock>;

      /**
       * @brief Get the singleton instance of the Initializer.
       * @return The singleton instance of the Initializer.
       */
      [[nodiscard]] static Initializer& getInstance()
      {
        static Initializer instance{};
        return instance;
      }

      /**
       * @brief Get the time stamp of the initialization.
       * @return The time stamp of the initialization.
       */
      [[nodiscard]] TimeStamp getTimeStamp()
      {
        return mTimeStamp;
      }

      void init();

      void finalize();

      /**
       * @brief Check if the library is initialized. Requires the mutex to be locked.
       * @return True if the library is initialized, false otherwise.
       */
      [[nodiscard]] bool isInitialized() const
      {
        return mTimeStamp != TimeStamp{};
      }
    private:
      /// @brief Default constructor.
      Initializer() = default;

      /// @brief Deleted copy constructor.
      Initializer(const Initializer&) = delete;

      /// @brief Deleted move constructor.
      Initializer(Initializer&&) = delete;

      /// @brief Destructor finalizing the library if it was initialized.
      ~Initializer()
      {
        finalize();
      }

      /// @brief Deleted copy assignment operator.
      Initializer& operator=(const Initializer&) = delete;

      /// @brief Deleted move assignment operator.
      Initializer& operator=(Initializer&&) = delete;

      TimeStamp mTimeStamp{}; ///< Time stamp of the initialization.
  };
} // namespace afft::detail

#ifdef AFFT_HEADER_ONLY

#ifdef AFFT_ENABLE_MPI
# include "mpi/init.hpp"
#endif

#ifdef AFFT_ENABLE_CLFFT
# include "clfft/init.hpp"
#endif
#ifdef AFFT_ENABLE_CUFFT
# include "cufft/init.hpp"
#endif
#ifdef AFFT_ENABLE_FFTW3
# include "fftw3/init.hpp"
#endif
#ifdef AFFT_ENABLE_HEFFTE
# include "heffte/init.hpp"
#endif
#ifdef AFFT_ENABLE_HIPFFT
# include "hipfft/init.hpp"
#endif
#ifdef AFFT_ENABLE_MKL
# include "mkl/init.hpp"
#endif
#ifdef AFFT_ENABLE_POCKETFFT
# include "pocketfft/init.hpp"
#endif
#ifdef AFFT_ENABLE_ROCFFT
# include "rocfft/init.hpp"
#endif
#ifdef AFFT_ENABLE_VKFFT
# include "vkfft/init.hpp"
#endif

namespace afft::detail
{
  AFFT_HEADER_ONLY_INLINE void Initializer::init()
  {
    if (isInitialized())
    {
      return;
    }

# ifdef AFFT_ENABLE_MPI
    mpi::init();
# endif

# ifdef AFFT_ENABLE_CUDA
    cuda::init();
# endif
# ifdef AFFT_ENABLE_HIP
    hip::init();
# endif
# ifdef AFFT_ENABLE_OPENCL
    opencl::init();
# endif

# ifdef AFFT_ENABLE_CLFFT
    clfft::init();
# endif
# ifdef AFFT_ENABLE_CUFFT
    cufft::init();
# endif
# ifdef AFFT_ENABLE_FFTW3
    fftw3::init();
# endif
# ifdef AFFT_ENABLE_HEFFTE
    heffte::init();
# endif
# ifdef AFFT_ENABLE_HIPFFT
    hipfft::init();
# endif
# ifdef AFFT_ENABLE_MKL
    mkl::init();
# endif
# ifdef AFFT_ENABLE_POCKETFFT
    pocketfft::init();
# endif
# ifdef AFFT_ENABLE_ROCFFT
    rocfft::init();
# endif
# ifdef AFFT_ENABLE_VKFFT
    vkfft::init();
# endif

    mTimeStamp = Clock::now();
  }

  AFFT_HEADER_ONLY_INLINE void Initializer::finalize()
  {
    if (!isInitialized())
    {
      return;
    }

# ifdef AFFT_ENABLE_CLFFT
    clfft::finalize();
# endif
# ifdef AFFT_ENABLE_CUFFT
    cufft::finalize();
# endif
# ifdef AFFT_ENABLE_FFTW3
    fftw3::finalize();
# endif
# ifdef AFFT_ENABLE_HEFFTE
    heffte::finalize();
# endif
# ifdef AFFT_ENABLE_HIPFFT
    hipfft::finalize();
# endif
# ifdef AFFT_ENABLE_MKL
    mkl::finalize();
# endif
# ifdef AFFT_ENABLE_POCKETFFT
    pocketfft::finalize();
# endif
# ifdef AFFT_ENABLE_ROCFFT
    rocfft::finalize();
# endif
# ifdef AFFT_ENABLE_VKFFT
    vkfft::finalize();
# endif

# ifdef AFFT_ENABLE_CUDA
    cuda::finalize();
# endif
# ifdef AFFT_ENABLE_HIP
    hip::finalize();
# endif
# ifdef AFFT_ENABLE_OPENCL
    opencl::finalize();
# endif

# ifdef AFFT_ENABLE_MPI
    mpi::finalize();
# endif

    mTimeStamp = TimeStamp{};
  }
} // namespace afft::detail

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_DETAIL_INIT_HPP */
