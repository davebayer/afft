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

/// @brief Macro for FFTW3 CPU transform backend
#define AFFT_CPU_TRANSFORM_BACKEND_FFTW3     (1 << 0)
/// @brief Macro for MKL CPU transform backend
#define AFFT_CPU_TRANSFORM_BACKEND_MKL       (1 << 1)
/// @brief Macro for PocketFFT CPU transform backend
#define AFFT_CPU_TRANSFORM_BACKEND_POCKETFFT (1 << 2)

/**
 * @brief Implementation of AFFT_CPU_TRANSFORM_BACKEND_FROM_NAME
 * @param backendName Name of the backend
 * @return Transform backend
 * @warning Do not use this macro directly
 */
#define AFFT_CPU_TRANSFORM_BACKEND_FROM_NAME_IMPL(backendName) \
  AFFT_CPU_TRANSFORM_BACKEND_##backendName

/**
 * @brief Macro for getting the transform backend from the name
 * @param backendName Name of the backend
 * @return Transform backend
 */
#define AFFT_CPU_TRANSFORM_BACKEND_FROM_NAME(backendName) \
  AFFT_CPU_TRANSFORM_BACKEND_FROM_NAME_IMPL(backendName)

/**
 * @brief Implementation of AFFT_CPU_TRANSFORM_BACKEND_MASK
 * @param ... List of transform backend names
 * @return Transform backend mask
 * @warning Do not use this macro directly
 */
#define AFFT_CPU_TRANSFORM_BACKEND_MASK_IMPL(...) \
  AFFT_BITOR(AFFT_FOR_EACH_WITH_DELIM(AFFT_CPU_TRANSFORM_BACKEND_FROM_NAME, AFFT_DELIM_COMMA, __VA_ARGS__))

/**
 * @brief Macro for getting the transform backend mask
 * @return Transform backend mask
 * @warning Requires AFFT_GPU_TRANSFORM_BACKEND_LIST to be defined
 */
#define AFFT_CPU_TRANSFORM_BACKEND_MASK \
  AFFT_CPU_TRANSFORM_BACKEND_MASK_IMPL(AFFT_CPU_TRANSFORM_BACKEND_LIST)

/**
 * @brief Macro for checking if the transform backend is allowed
 * @param backendName Name of the backend
 * @return Non zero if the transform backend is allowed, false otherwise
 */
#define AFFT_CPU_TRANSFORM_BACKEND_IS_ALLOWED(backendName) \
  (AFFT_CPU_TRANSFORM_BACKEND_FROM_NAME(backendName) & AFFT_CPU_TRANSFORM_BACKEND_MASK)

#include <complex>
#include <concepts>
#include <cstddef>
#include <memory>
#include <new>
#include <utility>

#include "common.hpp"

namespace afft::cpu
{
  /// @brief Enumeration of CPU transform backends
  enum class TransformBackend
  {
    fftw3     = AFFT_CPU_TRANSFORM_BACKEND_FFTW3,
    mkl       = AFFT_CPU_TRANSFORM_BACKEND_MKL,
    pocketfft = AFFT_CPU_TRANSFORM_BACKEND_POCKETFFT,
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
    Alignment alignment{alignments::defaultNew}; ///< Alignment for CPU memory allocation, defaults to `Alignment::defaultNew`
    unsigned  threadLimit{allThreads};           ///< Thread limit for CPU transform, 0 for no limit
  };

  /**
   * @brief Aligned memory deleter
   * @tparam T Type of the memory
   */
  template<typename T>
    requires (!std::same_as<T, void>)
  class AlignedDeleter
  {
    public:
      /// @brief Default constructor
      constexpr AlignedDeleter() noexcept = default;

      /// @brief Constructor with alignment
      constexpr AlignedDeleter(Alignment alignment) noexcept
      : mAlignment{alignment}
      {}

      /// @brief Copy constructor
      template<typename U>
        requires std::convertible_to<U*, T*>
      constexpr AlignedDeleter(const AlignedDeleter<U>& other) noexcept
      : mAlignment{other.mAlignment}
      {}

      /// @brief Move constructor
      template<typename U>
        requires std::convertible_to<U*, T*>
      constexpr AlignedDeleter(AlignedDeleter<U>&& other) noexcept
      : mAlignment{std::move(other.mAlignment)}
      {}

      /// @brief Destructor
      constexpr ~AlignedDeleter() noexcept = default;

      /// @brief Copy assignment operator
      template<typename U>
        requires std::convertible_to<U*, T*>
      constexpr AlignedDeleter& operator=(const AlignedDeleter<U>& other) noexcept
      {
        if (this != &other)
        {
          mAlignment = other.mAlignment;
        }
        return *this;
      }

      /// @brief Move assignment operator
      template<typename U>
        requires std::convertible_to<U*, T*>
      constexpr AlignedDeleter& operator=(AlignedDeleter<U>&& other) noexcept
      {
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
    public:
      /// @brief Default constructor
      constexpr AlignedDeleter() noexcept = default;

      /// @brief Constructor with alignment
      constexpr AlignedDeleter(Alignment alignment) noexcept
      : mAlignment{alignment}
      {}

      /// @brief Copy constructor
      template<typename U>
        requires std::convertible_to<U(*)[], T(*)[]>
      constexpr AlignedDeleter(const AlignedDeleter<U[]>& other) noexcept
      : mAlignment{other.mAlignment}
      {}

      /// @brief Move constructor
      template<typename U>
        requires std::convertible_to<U(*)[], T(*)[]>
      constexpr AlignedDeleter(AlignedDeleter<U[]>&& other) noexcept
      : mAlignment{std::move(other.mAlignment)}
      {}

      /// @brief Destructor
      constexpr ~AlignedDeleter() noexcept = default;

      /// @brief Copy assignment operator
      template<typename U>
        requires std::convertible_to<U(*)[], T(*)[]>
      constexpr AlignedDeleter& operator=(const AlignedDeleter<U[]>& other) noexcept
      {
        if (this != &other)
        {
          mAlignment = other.mAlignment;
        }
        return *this;
      }

      /// @brief Move assignment operator
      template<typename U>
        requires std::convertible_to<U(*)[], T(*)[]>
      constexpr AlignedDeleter& operator=(AlignedDeleter<U[]>&& other) noexcept
      {
        if (this != &other)
        {
          mAlignment = std::move(other.mAlignment);
        }
        return *this;
      }

      /// @brief Operator for deleting memory
      template<typename U>
        requires std::convertible_to<U(*)[], T(*)[]>
      void operator()(U* ptr) const
      {
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
  AlignedUniquePtr<T> makeAlignedUnique(Alignment alignment, Args&&... args)
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
  AlignedUniquePtr<T> makeAlignedUnique(Args&&... args)
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
    requires std::is_unbounded_array_v<T>
  AlignedUniquePtr<T> makeAlignedUnique(Alignment alignment, std::size_t n)
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
    requires std::is_unbounded_array_v<T>
  AlignedUniquePtr<T> makeAlignedUnique(std::size_t n)
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
  AlignedUniquePtr<T> makeAlignedUniqueForOverwrite(Alignment alignment)
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
  AlignedUniquePtr<T> makeAlignedUniqueForOverwrite()
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
    requires std::is_unbounded_array_v<T>
  AlignedUniquePtr<T> makeAlignedUniqueForOverwrite(Alignment alignment, std::size_t n)
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
    requires std::is_unbounded_array_v<T>
  AlignedUniquePtr<T> makeAlignedUniqueForOverwrite(std::size_t n)
  {
    return makeAlignedUniqueForOverwrite<T>(defaultAlignment, n);
  }

  /**
   * @class AlignedAllocator
   * @brief Allocator named concept implementation implementation for aligned CPU memory to be used with std::vector and
   *        others.
   * @tparam T Type of the memory
   */
  template<typename T>
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
      : mAlignment{other.mAlignment}
      {}

      /// @brief Move constructor
      template<typename U>
      constexpr AlignedAllocator(AlignedAllocator<U>&& other) noexcept
      : mAlignment{std::move(other.mAlignment)}
      {}

      /// @brief Destructor
      constexpr ~AlignedAllocator() noexcept = default;

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
      T* allocate(std::size_t n)
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
    protected:
    private:
      Alignment mAlignment{alignments::defaultNew}; ///< Alignment for memory allocation
  };
} // namespace afft::cpu

#endif /* AFFT_CPU_HPP */
