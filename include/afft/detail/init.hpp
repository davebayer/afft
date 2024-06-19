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

#include "architecture.hpp"
#include "../backend.hpp"

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
       * @brief Initialize the library.
       */
      void init()
      {
        // std::scoped_lock lock{mMutex};

        if (isInitialized())
        {
          return;
        }

        initImpl();
      }

      /**
       * @brief Finalize the library.
       */
      void finalize()
      {
        // std::unique_lock lock{mMutex};

        // mCondition.wait(lock, [this]() { return mActiveThreads != 0; });

        if (!isInitialized())
        {
          return;
        }

        finalizeImpl();
      }

      /**
       * @brief Get the time stamp of the initialization.
       * @return The time stamp of the initialization.
       */
      [[nodiscard]] TimeStamp getTimeStamp()
      {
        // std::scoped_lock lock{mMutex};

        return mTimeStamp;
      }

      /**
       * @brief Increment the number of active plans.
       */
      // void incrementActiveThreads()
      // {
      //   std::scoped_lock lock{mMutex};

      //   incrementActiveThreadsImpl();
      // }

      /**
       * @brief Decrement the number of active plans.
       */
      // void decrementActivePlans()
      // {
      //   std::scoped_lock lock{mMutex};

      //   decrementActiveThreadsImpl();
      // }
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
        // Do not lock the mutex here to avoid deadlocks.

        finalize();

        // Notify all waiting potentially blocking threads.
        // mCondition.notify_all();

        // Wait for all threads to finish.
        // std::unique_lock lock{mMutex};
      }

      /// @brief Deleted copy assignment operator.
      Initializer& operator=(const Initializer&) = delete;

      /// @brief Deleted move assignment operator.
      Initializer& operator=(Initializer&&) = delete;

      /**
       * @brief Check if the library is initialized. Requires the mutex to be locked.
       * @return True if the library is initialized, false otherwise.
       */
      [[nodiscard]] bool isInitialized() const
      {
        return mTimeStamp != TimeStamp{};
      }

      /**
       * @brief Initialize the library implementation. Requires the mutex to be locked.
       */
      void initImpl()
      {
#     ifdef AFFT_ENABLE_MPI
        mpi::init();
#     endif

#     if defined(AFFT_ENABLE_CUDA)
        cuda::init();
#     elif defined(AFFT_ENABLE_HIP)
        hip::init();
#     elif defined(AFFT_ENABLE_OPENCL)
        opencl::init();
#     endif

#     ifdef AFFT_ENABLE_CLFFT
        clfft::init();
#     endif
#     ifdef AFFT_ENABLE_CUFFT
        cufft::init();
#     endif
#     ifdef AFFT_ENABLE_FFTW3
        fftw3::init();
#     endif
#     ifdef AFFT_ENABLE_HEFFTE
        heffte::init();
#     endif
#     ifdef AFFT_ENABLE_HIPFFT
        hipfft::init();
#     endif
#     ifdef AFFT_ENABLE_MKL
        mkl::init();
#     endif
#     ifdef AFFT_ENABLE_POCKETFFT
        pocketfft::init();
#     endif
#     ifdef AFFT_ENABLE_ROCFFT
        rocfft::init();
#     endif
#     ifdef AFFT_ENABLE_VKFFT
        vkfft::init();
#     endif
      }

      /**
       * @brief Finalize the library implementation. Requires the mutex to be locked.
       */
      void finalizeImpl()
      {
#     ifdef AFFT_ENABLE_CLFFT
        clfft::finalize();
#     endif
#     ifdef AFFT_ENABLE_CUFFT
        cufft::finalize();
#     endif
#     ifdef AFFT_ENABLE_FFTW3
        fftw3::finalize();
#     endif
#     ifdef AFFT_ENABLE_HEFFTE
        heffte::finalize();
#     endif
#     ifdef AFFT_ENABLE_HIPFFT
        hipfft::finalize();
#     endif
#     ifdef AFFT_ENABLE_MKL
        mkl::finalize();
#     endif
#     ifdef AFFT_ENABLE_POCKETFFT
        pocketfft::finalize();
#     endif
#     ifdef AFFT_ENABLE_ROCFFT
        rocfft::finalize();
#     endif
#     ifdef AFFT_ENABLE_VKFFT
        vkfft::finalize();
#     endif

#     if defined(AFFT_ENABLE_CUDA)
        cuda::finalize();
#     elif defined(AFFT_ENABLE_HIP)
        hip::finalize();
#     elif defined(AFFT_ENABLE_OPENCL)
        opencl::finalize();
#     endif

#     ifdef AFFT_ENABLE_MPI
        mpi::finalize();
#     endif

        mTimeStamp   = TimeStamp{};
        // mActiveThreads = 0;
      }

      /**
       * @brief Increment the number of active threads implementation. Requires the mutex to be locked.
       */
      // void incrementActiveThreadsImpl()
      // {
      //   if (!isInitialized())
      //   {
      //     initImpl();
      //   }

      //   ++mActiveThreads;
      // }

      /**
       * @brief Decrement the number of active threads implementation. Requires the mutex to be locked.
       */
      // void decrementActiveThreadsImpl()
      // {
      //   if (mActiveThreads > 0)
      //   {
      //     --mActiveThreads;
      //   }

      //   if (mActiveThreads == 0)
      //   {
      //     mCondition.notify_all();
      //   }
      // }

      // std::mutex              mMutex{};         ///< Mutex.
      // std::atomic_size_t      mActiveThreads{}; ///< Disable finalize flag. 
      TimeStamp               mTimeStamp{};     ///< Time stamp of the initialization.
      // std::condition_variable mCondition{};     ///< Condition variable.
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_INIT_HPP */
