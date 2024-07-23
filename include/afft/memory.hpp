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

#ifndef AFFT_MEMORY_HPP
#define AFFT_MEMORY_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "common.hpp"
#include "Span.hpp"
#include "detail/cxx.hpp"

AFFT_EXPORT namespace afft
{
  /// @brief Memory layout type
  enum class MemoryLayout : ::afft_MemoryLayout
  {
    centralized = afft_MemoryLayout_centralized, ///< Centralized memory layout, only when the transformation is executed by single process on single target
    distributed = afft_MemoryLayout_distributed, ///< Distributed memory layout, for distributed transformations over multiple processes or targets
  };

  /// @brief Alignment of a data type
  enum class Alignment : ::afft_Alignment
  {
    defaultNew = __STDCPP_DEFAULT_NEW_ALIGNMENT__, ///< Default alignment for new operator
    simd128    = afft_Alignment_simd128,           ///< 128-bit SIMD alignment
    simd256    = afft_Alignment_simd256,           ///< 256-bit SIMD alignment
    simd512    = afft_Alignment_simd512,           ///< 512-bit SIMD alignment
    simd1024   = afft_Alignment_simd1024,          ///< 1024-bit SIMD alignment
    simd2048   = afft_Alignment_simd2048,          ///< 2048-bit SIMD alignment

    sse        = afft_Alignment_sse,               ///< SSE alignment
    sse2       = afft_Alignment_sse2,              ///< SSE2 alignment
    sse3       = afft_Alignment_sse3,              ///< SSE3 alignment
    sse4       = afft_Alignment_sse4,              ///< SSE4 alignment
    sse4_1     = afft_Alignment_sse4_1,            ///< SSE4.1 alignment
    sse4_2     = afft_Alignment_sse4_2,            ///< SSE4.2 alignment
    avx        = afft_Alignment_avx,               ///< AVX alignment
    avx2       = afft_Alignment_avx2,              ///< AVX2 alignment
    avx512     = afft_Alignment_avx512,            ///< AVX-512 alignment
    neon       = afft_Alignment_neon,              ///< NEON alignment
    sve        = afft_Alignment_sve,               ///< SVE alignment

    cpuNative  = afft_Alignment_cpuNative,         ///< cpu native alignment
  };

  /// @brief Complex number format
  enum class ComplexFormat : ::afft_ComplexFormat
  {
    interleaved = afft_ComplexFormat_interleaved, ///< interleaved complex format
    planar      = afft_ComplexFormat_planar,      ///< planar complex format
  };
  
  /// @brief Memory layout of the centralized transform
  struct CentralizedMemoryLayout
  {
    Alignment     alignment{};                               ///< alignment of the memory
    ComplexFormat complexFormat{ComplexFormat::interleaved}; ///< complex number format
    View<Size>    srcStrides{};                              ///< strides of the source memory
    View<Size>    dstStrides{};                              ///< strides of the destination memory
  };

  /// @brief Memory block of the distributed transform
  struct MemoryBlock
  {
    View<Size> starts{};  ///< starts of the memory block
    View<Size> sizes{};   ///< sizes of the memory block
    View<Size> strides{}; ///< strides of the memory block
  };

  /// @brief Memory layout of the distributed transform
  struct DistributedMemoryLayout
  {
    Alignment         alignment{};                               ///< alignment of the memory
    ComplexFormat     complexFormat{ComplexFormat::interleaved}; ///< complex number format
    View<MemoryBlock> srcBlocks{};                               ///< source memory blocks
    View<Axis>        srcDistribAxes{};                          ///< axes along which the source data are distributed
    View<Axis>        srcAxesOrder{};                            ///< order of the source axes
    View<MemoryBlock> dstBlocks{};                               ///< destination memory blocks
    View<Axis>        dstDistribAxes{};                          ///< axes along which the destination data are distributed
    View<Axis>        dstAxesOrder{};                            ///< order of the destination axes
  };

  /**
   * @brief Aligned memory deleter
   * @tparam T Type of the memory
   */
  template<typename T>
  class AlignedDeleter
  {
    static_assert(std::is_object_v<T>, "T must be an object type");

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
      Alignment mAlignment{Alignment::defaultNew}; ///< Alignment for memory allocation
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
      Alignment mAlignment{Alignment::defaultNew}; ///< Alignment for memory allocation
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
    return makeAlignedUnique<T>(Alignment::defaultNew, std::forward<Args>(args)...);
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
    return makeAlignedUnique<T>(Alignment::defaultNew, n);
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
    return makeAlignedUniqueForOverwrite<T>(Alignment::defaultNew);
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
    return makeAlignedUniqueForOverwrite<T>(Alignment::defaultNew, n);
  }

  /**
   * @class AlignedAllocator
   * @brief Allocator named concept implementation implementation for aligned memory to be used with std::vector and
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
      Alignment mAlignment{Alignment::defaultNew}; ///< Alignment for memory allocation
  };

  namespace c
  {
    using MemoryLayout            = ::afft_MemoryLayout;
    using Alignment               = ::afft_Alignment;
    using ComplexFormat           = ::afft_ComplexFormat;
    using CentralizedMemoryLayout = ::afft_CentralizedMemoryLayout;
    using MemoryBlock             = ::afft_MemoryBlock;
    using DistributedMemoryLayout = ::afft_DistributedMemoryLayout;

    /**
     * @brief Allocate aligned memory
     * @param sizeInBytes Size of the memory in bytes
     * @param alignment Alignment of the memory
     * @return Pointer to the allocated memory
     */
    [[nodiscard]] inline void* alignedAlloc(std::size_t sizeInBytes, Alignment alignment) noexcept
    {
      return ::afft_alignedAlloc(sizeInBytes, alignment);
    }

    /**
     * @brief Free aligned memory
     * @param ptr Pointer to the memory
     * @param alignment Alignment of the memory
     */
    inline void alignedFree(void* ptr, Alignment alignment) noexcept
    {
      ::afft_alignedFree(ptr, alignment);
    }
  } // namespace c
} // namespace afft

#endif /* AFFT_MEMORY_HPP */
