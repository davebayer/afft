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

#ifndef AFFT_CPU_HPP
#define AFFT_CPU_HPP

#include "macro.hpp"

/// @brief Macro for FFTW3 CPU backend
#define AFFT_CPU_BACKEND_FFTW3     (1 << 0)
/// @brief Macro for MKL CPU backend
#define AFFT_CPU_BACKEND_MKL       (1 << 1)
/// @brief Macro for PocketFFT CPU backend
#define AFFT_CPU_BACKEND_POCKETFFT (1 << 2)

/**
 * @brief Macro for getting the backend from the name
 * @param backendName Name of the backend
 * @return Transform backend
 */
#define AFFT_CPU_BACKEND_FROM_NAME(backendName) \
  AFFT_DETAIL_EXPAND_AND_CONCAT(AFFT_CPU_BACKEND_, backendName)

/**
 * @brief Macro for checking if the backend is allowed
 * @param backendName Name of the backend
 * @return Non zero if the backend is allowed, false otherwise
 */
#define AFFT_CPU_BACKEND_IS_ENABLED(backendName) \
  (AFFT_CPU_BACKEND_FROM_NAME(backendName) & (AFFT_CPU_BACKEND_MASK))

#include <array>
#include <complex>
#include <concepts>
#include <cstddef>
#include <memory>
#include <new>
#include <string_view>
#include <utility>

#include "common.hpp"
#include "detail/cxx.hpp"

namespace afft::cpu
{
  /// @brief Enumeration of CPU backends
  enum class Backend
  {
    fftw3,
    mkl,
    pocketfft,
  };

  /// @brief Number of CPU backends
  inline constexpr std::size_t backendCount{3};

  namespace fftw3
  {
    /// @brief Init parameters for FFTW3 CPU backend
    struct InitParameters
    {
      std::string_view floatWisdom{};      ///< Wisdom for single precision (fftwf)
      std::string_view doubleWisdom{};     ///< Wisdom for double precision (fftw)
      std::string_view longDoubleWisdom{}; ///< Wisdom for long double precision (fftwl)
      std::string_view quadWisdom{};       ///< Wisdom for quad precision (fftwq)
    };
  } // namespace fftw3

  namespace mkl
  {
    /// @brief Init parameters for MKL CPU backend
    struct InitParameters {};
  } // namespace mkl

  namespace pocketfft
  {
    /// @brief Init parameters for PocketFFT CPU backend
    struct InitParameters {};
  } // namespace pocketfft

  /// @brief Init parameters for CPU backends
  struct InitParameters
  {
    fftw3::InitParameters     fftw3{};     ///< FFTW3 init parameters
    mkl::InitParameters       mkl{};       ///< MKL init parameters
    pocketfft::InitParameters pocketfft{}; ///< PocketFFT init parameters
  };

  /// @brief alignments for CPU memory allocation
  namespace alignments
  {
    inline constexpr Alignment defaultNew{__STDCPP_DEFAULT_NEW_ALIGNMENT__}; ///< Default alignment for new operator
    inline constexpr Alignment simd128{16};                                  ///< 128-bit SIMD alignment
    inline constexpr Alignment simd256{32};                                  ///< 256-bit SIMD alignment
    inline constexpr Alignment simd512{64};                                  ///< 512-bit SIMD alignment

    inline constexpr Alignment sse{simd128};                                 ///< SSE alignment
    inline constexpr Alignment sse2{simd128};                                ///< SSE2 alignment
    inline constexpr Alignment sse3{simd128};                                ///< SSE3 alignment
    inline constexpr Alignment sse4{simd128};                                ///< SSE4 alignment
    inline constexpr Alignment sse4_1{simd128};                              ///< SSE4.1 alignment
    inline constexpr Alignment sse4_2{simd128};                              ///< SSE4.2 alignment
    inline constexpr Alignment avx{simd256};                                 ///< AVX alignment
    inline constexpr Alignment avx2{simd256};                                ///< AVX2 alignment
    inline constexpr Alignment avx512{simd512};                              ///< AVX-512 alignment
    inline constexpr Alignment neon{simd128};                                ///< NEON alignment
  } // namespace Alignment

#if defined(__AVX512F__)
  inline constexpr auto defaultAlignment = alignments::avx512;
#elif defined(__AVX2__)
  inline constexpr auto defaultAlignment = alignments::avx2;
#elif defined(__AVX__)
  inline constexpr auto defaultAlignment = alignments::avx;
#elif defined(__SSE4_2__)
  inline constexpr auto defaultAlignment = alignments::sse4_2;
#elif defined(__SSE4_1__)
  inline constexpr auto defaultAlignment = alignments::sse4_1;
#elif defined(__SSE4__)
  inline constexpr auto defaultAlignment = alignments::sse4;
#elif defined(__SSE3__)
  inline constexpr auto defaultAlignment = alignments::sse3;
#elif defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP == 2)
  inline constexpr auto defaultAlignment = alignments::sse2;
#elif defined(__SSE__) || (defined(_M_IX86_FP) && _M_IX86_FP == 1)
  inline constexpr auto defaultAlignment = alignments::sse;
#elif defined(__ARM_NEON) || defined(_M_ARM_NEON)
  inline constexpr auto defaultAlignment = alignments::neon;
#else
  inline constexpr auto defaultAlignment = alignments::defaultNew;
#endif

  inline constexpr unsigned allThreads{}; ///< All threads for CPU transform

  /**
   * @struct Parameters
   * @brief Parameters for CPU transform
   */
  struct Parameters
  {
    MemoryLayout memoryLayout{};                    ///< Memory layout for CPU transform
    Alignment    alignment{alignments::defaultNew}; ///< Alignment for CPU memory allocation, defaults to `alignments::defaultNew`
    unsigned     threadLimit{allThreads};           ///< Thread limit for CPU transform, 0 for no limit
  };

  /// @brief Default list of backends
  inline constexpr std::array defaultBackendList
  {
    Backend::mkl,       // prefer mkl
    Backend::fftw3,     // if mkl cannot create plan, fallback fftw3
    Backend::pocketfft, // fallback to pocketfft
  };

  /// @brief Set up strategy for backend selection
  struct BackendSelectParameters
  {
    Span<const Backend>   backends{defaultBackendList};           ///< Priority of the backends
    Span<std::string>     backendsErrors{};                       ///< Why a backend creation failed
    BackendSelectStrategy strategy{BackendSelectStrategy::first}; ///< Select strategy
  };

  /// @brief Execution parameters for CPU transform
  struct ExecutionParameters {};

  /**
   * @brief Aligned memory deleter
   * @tparam T Type of the memory
   */
  template<typename T>
  class AlignedDeleter
  {
    static_assert(!std::is_void_v<T>, "T cannot be void");

    public:
      /// @brief Default constructor
      constexpr AlignedDeleter() noexcept = default;

      /// @brief Constructor with alignment
      constexpr AlignedDeleter(Alignment alignment) noexcept
      : mAlignment{alignment}
      {}

      /// @brief Copy constructor
      template<typename U>
      constexpr AlignedDeleter(const AlignedDeleter<U>& other) noexcept
      : mAlignment{other.mAlignment}
      {
        static_assert(std::is_convertible_v<U*, T*>, "U* must be convertible to T*");
      }

      /// @brief Move constructor
      template<typename U>
      constexpr AlignedDeleter(AlignedDeleter<U>&& other) noexcept
      : mAlignment{std::move(other.mAlignment)}
      {
        static_assert(std::is_convertible_v<U*, T*>, "U* must be convertible to T*");
      }

      /// @brief Destructor
      ~AlignedDeleter() noexcept = default;

      /// @brief Copy assignment operator
      template<typename U>
      constexpr AlignedDeleter& operator=(const AlignedDeleter<U>& other) noexcept
      {
        static_assert(std::is_convertible_v<U*, T*>, "U* must be convertible to T*");

        if (this != &other)
        {
          mAlignment = other.mAlignment;
        }
        return *this;
      }

      /// @brief Move assignment operator
      template<typename U>
      constexpr AlignedDeleter& operator=(AlignedDeleter<U>&& other) noexcept
      {
        static_assert(std::is_convertible_v<U*, T*>, "U* must be convertible to T*");

        if (this != &other)
        {
          mAlignment = std::move(other.mAlignment);
        }
        return *this;
      }
      
      /// @brief Operator for deleting memory
      void operator()(T* ptr) const
      {
        static_assert(sizeof(T) > 0, "T must be a complete type");

        if (ptr != nullptr)
        {
          ::operator delete(ptr, static_cast<std::align_val_t>(mAlignment));
        }
      }
    private:
      Alignment mAlignment{defaultAlignment}; ///< Alignment for memory allocation
  };

  /**
   * @brief Specialization of AlignedDeleter for arrays
   * @tparam T Type of the memory
   */
  template<typename T>
  class AlignedDeleter<T[]>
  {
    static_assert(!std::is_void_v<T>, "T cannot be void");

    public:
      /// @brief Default constructor
      constexpr AlignedDeleter() noexcept = default;

      /// @brief Constructor with alignment
      constexpr AlignedDeleter(Alignment alignment) noexcept
      : mAlignment{alignment}
      {}

      /// @brief Copy constructor
      template<typename U>
      constexpr AlignedDeleter(const AlignedDeleter<U[]>& other) noexcept
      : mAlignment{other.mAlignment}
      {
        static_assert(std::is_convertible_v<U(*)[], T(*)[]>, "U(*)[] must be convertible to T(*)[]");
      }

      /// @brief Move constructor
      template<typename U>
      constexpr AlignedDeleter(AlignedDeleter<U[]>&& other) noexcept
      : mAlignment{std::move(other.mAlignment)}
      {
        static_assert(std::is_convertible_v<U(*)[], T(*)[]>, "U(*)[] must be convertible to T(*)[]");
      }

      /// @brief Destructor
      ~AlignedDeleter() noexcept = default;

      /// @brief Copy assignment operator
      template<typename U>
      constexpr AlignedDeleter& operator=(const AlignedDeleter<U[]>& other) noexcept
      {
        static_assert(std::is_convertible_v<U(*)[], T(*)[]>, "U(*)[] must be convertible to T(*)[]");

        if (this != &other)
        {
          mAlignment = other.mAlignment;
        }
        return *this;
      }

      /// @brief Move assignment operator
      template<typename U>
      constexpr AlignedDeleter& operator=(AlignedDeleter<U[]>&& other) noexcept
      {
        static_assert(std::is_convertible_v<U(*)[], T(*)[]>, "U(*)[] must be convertible to T(*)[]");

        if (this != &other)
        {
          mAlignment = std::move(other.mAlignment);
        }
        return *this;
      }

      /// @brief Operator for deleting memory
      template<typename U>
      void operator()(U* ptr) const
      {
        static_assert(std::is_convertible_v<U(*)[], T(*)[]>, "U(*)[] must be convertible to T(*)[]");
        static_assert(sizeof(T) > 0, "T must be a complete type");

        if (ptr != nullptr)
        {
          ::operator delete[](ptr, static_cast<std::align_val_t>(mAlignment));
        }
      }
    private:
      Alignment mAlignment{defaultAlignment}; ///< Alignment for memory allocation
  };

  /**
   * @brief Alias for aligned unique pointer
   * @tparam T Type of the memory
   */
  template<typename T>
  using AlignedUniquePtr = std::unique_ptr<T, AlignedDeleter<T>>;

  /**
   * @brief Make aligned unique pointer
   * @tparam T Type of the memory
   * @tparam Args Types of the arguments
   * @param alignment Alignment for memory allocation
   * @param args Arguments for the constructor
   * @return Aligned unique pointer
   */
  template<typename T, typename... Args>
  [[nodiscard]] auto makeAlignedUnique(Alignment alignment, Args&&... args)
    -> AFFT_RET_REQUIRES(AlignedUniquePtr<T>, !detail::cxx::is_unbounded_array_v<T>)
  {
    const auto align = static_cast<std::align_val_t>(alignment);

    return AlignedUniquePtr<T>(new(align) T{std::forward<Args>(args)...}, AlignedDeleter<T>{alignment});
  }
  
  /**
   * @brief Make aligned unique pointer
   * @tparam T Type of the memory
   * @tparam Args Types of the arguments
   * @param args Arguments for the constructor
   * @return Aligned unique pointer
   */
  template<typename T, typename... Args>
  [[nodiscard]] auto makeAlignedUnique(Args&&... args)
    -> AFFT_RET_REQUIRES(AlignedUniquePtr<T>, !detail::cxx::is_unbounded_array_v<T>)
  {
    return makeAlignedUnique<T>(defaultAlignment, std::forward<Args>(args)...);
  }

  /**
   * @brief Make aligned unique pointer for arrays
   * @tparam T Type of the memory
   * @param alignment Alignment for memory allocation
   * @param n Number of elements
   * @return Aligned unique pointer
   */
  template<typename T>
  [[nodiscard]] auto makeAlignedUnique(Alignment alignment, std::size_t n)
    -> AFFT_RET_REQUIRES(AlignedUniquePtr<T>, detail::cxx::is_unbounded_array_v<T>)
  {
    using U = std::remove_extent_t<T>;

    const auto align = static_cast<std::align_val_t>(alignment);

    return AlignedUniquePtr<T>(new(align) U[n]{}, AlignedDeleter<T>{alignment});
  }

  /**
   * @brief Make aligned unique pointer for arrays
   * @tparam T Type of the memory
   * @param n Number of elements
   * @return Aligned unique pointer
   */
  template<typename T>
  [[nodiscard]] auto makeAlignedUnique(std::size_t n)
    -> AFFT_RET_REQUIRES(AlignedUniquePtr<T>, detail::cxx::is_unbounded_array_v<T>)
  {
    return makeAlignedUnique<T>(defaultAlignment, n);
  }

  /**
   * @brief Make aligned unique pointer to be overwritten
   * @tparam T Type of the memory
   * @param alignment Alignment for memory allocation
   * @return Aligned unique pointer
   */
  template<typename T>
  [[nodiscard]] auto makeAlignedUniqueForOverwrite(Alignment alignment)
    -> AFFT_RET_REQUIRES(AlignedUniquePtr<T>, !detail::cxx::is_unbounded_array_v<T>)
  {
    const auto align = static_cast<std::align_val_t>(alignment);

    return AlignedUniquePtr<T>(new(align) T, AlignedDeleter<T>{alignment});
  }

  /**
   * @brief Make aligned unique pointer to be overwritten
   * @tparam T Type of the memory
   * @return Aligned unique pointer
   */
  template<typename T>
  [[nodiscard]] auto makeAlignedUniqueForOverwrite()
    -> AFFT_RET_REQUIRES(AlignedUniquePtr<T>, !detail::cxx::is_unbounded_array_v<T>)
  {
    return makeAlignedUniqueForOverwrite<T>(defaultAlignment);
  }

  /**
   * @brief Make aligned unique pointer for arrays to be overwritten
   * @tparam T Type of the memory
   * @param alignment Alignment for memory allocation
   * @param n Number of elements
   * @return Aligned unique pointer
   */
  template<typename T>
  [[nodiscard]] auto makeAlignedUniqueForOverwrite(Alignment alignment, std::size_t n)
    -> AFFT_RET_REQUIRES(AlignedUniquePtr<T>, detail::cxx::is_unbounded_array_v<T>)
  {
    using U = std::remove_extent_t<T>;

    const auto align = static_cast<std::align_val_t>(alignment);

    return AlignedUniquePtr<T>(new(align) U[n], AlignedDeleter<T>{alignment});
  }

  /**
   * @brief Make aligned unique pointer for arrays to be overwritten
   * @tparam T Type of the memory
   * @param n Number of elements
   * @return Aligned unique pointer
   */
  template<typename T>
  [[nodiscard]] auto makeAlignedUniqueForOverwrite(std::size_t n)
    -> AFFT_RET_REQUIRES(AlignedUniquePtr<T>, detail::cxx::is_unbounded_array_v<T>)
  {
    return makeAlignedUniqueForOverwrite<T>(defaultAlignment, n);
  }

  /**
   * @class AlignedAllocator
   * @brief Allocator named concept implementation implementation for aligned CPU memory to be used with std::vector and
   *        others.
   * @tparam T Type of the memory
   */
  template<typename T = void>
  class AlignedAllocator
  {
    public:
      /// @brief Type of the memory
      using value_type = T;

      /// @brief Default constructor
      constexpr AlignedAllocator() = default;

      /// @brief Constructor with alignment
      constexpr AlignedAllocator(Alignment alignment) noexcept
      : mAlignment{alignment}
      {}

      /// @brief Copy constructor
      template<typename U>
      constexpr AlignedAllocator(const AlignedAllocator<U>& other) noexcept
      : mAlignment{other.getAlignment()}
      {}

      /// @brief Move constructor
      template<typename U>
      constexpr AlignedAllocator(AlignedAllocator<U>&& other) noexcept
      : mAlignment{std::move(other.mAlignment)}
      {}

      /// @brief Destructor
      ~AlignedAllocator() noexcept = default;

      /// @brief Copy assignment operator
      template<typename U>
      constexpr AlignedAllocator& operator=(const AlignedAllocator<U>& other) noexcept
      {
        if (this != &other)
        {
          mAlignment = other.mAlignment;
        }
        return *this;
      }

      /// @brief Move assignment operator
      template<typename U>
      constexpr AlignedAllocator& operator=(AlignedAllocator<U>&& other) noexcept
      {
        if (this != &other)
        {
          mAlignment = std::move(other.mAlignment);
        }
        return *this;
      }

      /**
       * @brief Allocate memory
       * @param n Number of elements
       * @return Pointer to the allocated memory
       */
      [[nodiscard]] T* allocate(std::size_t n)
      {
        return static_cast<T*>(::operator new(n * sizeof(T), static_cast<std::align_val_t>(mAlignment)));
      }

      /**
       * @brief Deallocate memory
       * @param p Pointer to the memory
       * @param n Number of elements
       */
      void deallocate(T* p, std::size_t) noexcept
      {
        ::operator delete(p, static_cast<std::align_val_t>(mAlignment));
      }

      /**
       * @brief Get alignment
       * @return Alignment
       */
      [[nodiscard]] constexpr Alignment getAlignment() const noexcept
      {
        return mAlignment;
      }
    protected:
    private:
      Alignment mAlignment{alignments::defaultNew}; ///< Alignment for memory allocation
  };
} // namespace afft::cpu

#endif /* AFFT_CPU_HPP */
