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
# include <afft/detail/include.hpp>
#endif

#include <afft/common.hpp>
#include <afft/error.hpp>
#include <afft/detail/cxx.hpp>
#include <afft/detail/utils.hpp>

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
    Alignment     alignment{Alignment::defaultNew};          ///< alignment of the memory
    ComplexFormat complexFormat{ComplexFormat::interleaved}; ///< complex number format
    const Stride* srcStrides{};                              ///< Source strides (null for default or array of size shapeRank)
    const Stride* dstStrides{};                              ///< Destination strides (null for default or array of size shapeRank)
  };

  /// @brief Memory layout of the distributed transform
  struct DistributedMemoryLayout
  {
    Alignment            alignment{Alignment::defaultNew};          ///< alignment of the memory
    ComplexFormat        complexFormat{ComplexFormat::interleaved}; ///< complex number format
    const Axis*          srcDistribAxes{};                          ///< axes along which the source data are distributed
    std::size_t          srcDistribAxesRank{};                      ///< rank of the source distributed axes
    const Axis*          srcAxesOrder{};                            ///< order of the source axes
    const Size* const*   srcStarts{};                               ///< starting indices of the source memory
    const Size* const*   srcSizes{};                                ///< sizes of the source memory
    const Stride* const* srcStrides{};                              ///< strides of the source memory
    const Axis*          dstDistribAxes{};                          ///< axes along which the destination data are distributed
    std::size_t          dstDistribAxesRank{};                      ///< rank of the destination distributed axes
    const Axis*          dstAxesOrder{};                            ///< order of the destination axes
    const Size* const*   dstStarts{};                               ///< starting indices of the destination memory
    const Size* const*   dstSizes{};                                ///< sizes of the destination memory
    const Stride* const* dstStrides{};                              ///< strides of the destination memory
  };

  /// @brief Memory layout variant
  using MemoryLayoutParametersVariant = std::variant<std::monostate, CentralizedMemoryLayout, DistributedMemoryLayout>;

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
   * @param minAlignment Minimal alignment for memory allocation
   * @param args Arguments for the constructor
   * @return Aligned unique pointer
   */
  template<typename T, typename... Args>
  [[nodiscard]] auto makeAlignedUnique(Alignment minAlignment, Args&&... args)
    -> AFFT_RET_REQUIRES(AlignedUniquePtr<T>, !detail::cxx::is_unbounded_array_v<T>)
  {
    const std::size_t alignment = std::max(static_cast<std::size_t>(minAlignment), alignof(T));

    T* ptr = static_cast<T*>(::operator new(sizeof(T), std::align_val_t{alignment}));

    return AlignedUniquePtr<T>{::new(ptr) T{std::forward<Args>(args)...}, AlignedDeleter<T>{Alignment{alignment}}};
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
   * @param minAlignment Minimal alignment for memory allocation
   * @param n Number of elements
   * @return Aligned unique pointer
   */
  template<typename T>
  [[nodiscard]] auto makeAlignedUnique(Alignment minAlignment, std::size_t n)
    -> AFFT_RET_REQUIRES(AlignedUniquePtr<T>, detail::cxx::is_unbounded_array_v<T>)
  {
    using U = std::remove_extent_t<T>;

    const std::size_t alignment = std::max(static_cast<std::size_t>(minAlignment), alignof(U));

    U* ptr = static_cast<U*>(::operator new[](n * sizeof(U), std::align_val_t{alignment}));

    return AlignedUniquePtr<T>{::new(ptr) U[n]{}, AlignedDeleter<T>{Alignment{alignment}}};
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
   * @brief Make aligned unique pointer for bounded arrays (deleted)
   * @tparam T Type of the memory
   * @tparam Args Types of the arguments
   * @param args Arguments for the constructor
   * @return Aligned unique pointer
   */
  template<typename T, typename... Args>
  [[nodiscard]] auto makeAlignedUnique(Args&&... args)
    -> AFFT_RET_REQUIRES(AlignedUniquePtr<T>, detail::cxx::is_bounded_array_v<T>) = delete;

  /**
   * @brief Make aligned unique pointer to be overwritten
   * @tparam T Type of the memory
   * @param minAlignment Minimal alignment for memory allocation
   * @return Aligned unique pointer
   */
  template<typename T>
  [[nodiscard]] auto makeAlignedUniqueForOverwrite(Alignment minAlignment)
    -> AFFT_RET_REQUIRES(AlignedUniquePtr<T>, !detail::cxx::is_unbounded_array_v<T>)
  {
    const std::size_t alignment = std::max(static_cast<std::size_t>(minAlignment), alignof(T));

    T* ptr = static_cast<T*>(::operator new(sizeof(T), std::align_val_t{alignment}));

    return AlignedUniquePtr<T>{::new(ptr) T, AlignedDeleter<T>{Alignment{alignment}}};
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
   * @param minAlignment Minimal alignment for memory allocation
   * @param n Number of elements
   * @return Aligned unique pointer
   */
  template<typename T>
  [[nodiscard]] auto makeAlignedUniqueForOverwrite(Alignment minAlignment, std::size_t n)
    -> AFFT_RET_REQUIRES(AlignedUniquePtr<T>, detail::cxx::is_unbounded_array_v<T>)
  {
    using U = std::remove_extent_t<T>;

    const std::size_t alignment = std::max(static_cast<std::size_t>(minAlignment), alignof(U));

    U* ptr = static_cast<U*>(::operator new[](n * sizeof(U), std::align_val_t{alignment}));

    return AlignedUniquePtr<T>{::new(ptr) U[n], AlignedDeleter<T>{Alignment{alignment}}};
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
   * @brief Make aligned unique pointer for bounded arrays to be overwritten (deleted)
   * @tparam T Type of the memory
   * @param minAlignment Minimal alignment for memory allocation
   * @param n Number of elements
   * @return Aligned unique pointer
   */
  template<typename T>
  [[nodiscard]] auto makeAlignedUniqueForOverwrite(Alignment minAlignment, std::size_t n)
    -> AFFT_RET_REQUIRES(AlignedUniquePtr<T>, detail::cxx::is_bounded_array_v<T>) = delete;

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

      /**
       * @brief Constructor with minimal alignment
       * @param minAlignment Minimal alignment for memory allocation
       */
      constexpr AlignedAllocator(Alignment minAlignment) noexcept
      : mAlignment{std::max(static_cast<std::size_t>(minAlignment), alignof(T))}
      {}

      /// @brief Copy constructor
      template<typename U>
      constexpr AlignedAllocator(const AlignedAllocator<U>& other) noexcept
      : mAlignment{std::max(static_cast<std::size_t>(other.getAlignment()), alignof(T))}
      {}

      /// @brief Move constructor
      template<typename U>
      constexpr AlignedAllocator(AlignedAllocator<U>&& other) noexcept
      : mAlignment{std::max(static_cast<std::size_t>(other.getAlignment()), alignof(T))}
      {}

      /// @brief Destructor
      ~AlignedAllocator() noexcept = default;

      /// @brief Copy assignment operator
      template<typename U>
      constexpr AlignedAllocator& operator=(const AlignedAllocator<U>& other) noexcept
      {
        if (this != &other)
        {
          mAlignment = Alignment{std::max(static_cast<std::size_t>(other.getAlignment()), alignof(T))};
        }
        return *this;
      }

      /// @brief Move assignment operator
      template<typename U>
      constexpr AlignedAllocator& operator=(AlignedAllocator<U>&& other) noexcept
      {
        if (this != &other)
        {
          mAlignment = Alignment{std::max(static_cast<std::size_t>(other.getAlignment()), alignof(T))};
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

  /**
   * @brief Get the alignment of the pointers
   * @tparam PtrTs Pointer types
   * @param ptrs Pointers
   * @return Alignment
   */
  template<typename... PtrTs>
  [[nodiscard]] Alignment alignmentOf(const PtrTs*... ptrs) noexcept
  {
    static_assert(sizeof...(ptrs) > 0, "At least one pointer must be provided");

    const auto bitOredPtrs = (0 | ... | reinterpret_cast<detail::cxx::uintptr_t>(ptrs));

    return static_cast<Alignment>(bitOredPtrs & ~(bitOredPtrs - 1));
  }

  /**
   * @brief Make strides.
   * @param shape Shape
   * @param shapeRank Shape rank
   * @param fastestAxisStride Stride of the fastest axis
   * @param strides Strides
   */
  template<typename StrideT = Stride, typename ShapeT>
  constexpr void makeStrides(const ShapeT*     shape,
                             const std::size_t shapeRank,
                             StrideT*          strides,
                             const StrideT     fastestAxisStride = 1)
  {
    static_assert(std::is_integral_v<StrideT>, "StrideT must be an integral type");
    static_assert(std::is_integral_v<ShapeT>, "ShapeT must be an integral type");

    if (shapeRank == 0)
    {
      throw Exception{Error::invalidArgument, "shape rank must be greater than zero"};
    }

    if (shape == nullptr)
    {
      throw Exception{Error::invalidArgument, "invalid shape"};
    }

    if (strides == nullptr)
    {
      throw Exception{Error::invalidArgument, "invalid strides"};
    }

    if (fastestAxisStride == 0)
    {
      throw Exception{Error::invalidArgument, "fastest axis stride must be greater than zero"};
    }

    if (detail::cxx::any_of(shape, shape + shapeRank, detail::IsZero<ShapeT>{}))
    {
      throw Exception{Error::invalidArgument, "shape must not contain zeros"};
    }

    strides[shapeRank - 1] = fastestAxisStride;

    for (std::size_t i = shapeRank - 1; i > 0; --i)
    {
      strides[i - 1] = detail::safeIntCast<StrideT>(shape[i]) * strides[i];
    }
  }

  /**
   * @brief Make transposed strides.
   * @param shape Shape
   * @param shapeRank Shape rank
   * @param orgAxesOrder Original axes order
   * @param strides Strides
   * @param fastestAxisStride Stride of the fastest axis
   */
  template<typename StrideT = Stride, typename ShapeT, typename AxisT>
  inline void makeTransposedStrides(const ShapeT*     shape,
                                    const std::size_t shapeRank,
                                    const AxisT*      orgAxesOrder,
                                    StrideT*          strides,
                                    const StrideT     fastestAxisStride = 1)
  {
    static_assert(std::is_integral_v<StrideT>, "StrideT must be an integral type");
    static_assert(std::is_integral_v<ShapeT>, "ShapeT must be an integral type");
    static_assert(std::is_integral_v<AxisT>, "AxisT must be an integral type");

    if (shapeRank == 0)
    {
      throw Exception{Error::invalidArgument, "shape rank must be greater than zero"};
    }

    if (shape == nullptr)
    {
      throw Exception{Error::invalidArgument, "invalid shape"};
    }

    if (orgAxesOrder == nullptr)
    {
      throw Exception{Error::invalidArgument, "invalid axes order"};
    }

    if (strides == nullptr)
    {
      throw Exception{Error::invalidArgument, "invalid strides"};
    }

    if (fastestAxisStride == 0)
    {
      throw Exception{Error::invalidArgument, "fastest axis stride must be greater than zero"};
    }

    if (detail::cxx::any_of(shape, shape + shapeRank, detail::IsZero<ShapeT>{}))
    {
      throw Exception{Error::invalidArgument, "shape must not contain zeros"};
    }

    // Check if axes order is valid
    {
      std::bitset<maxRank> seenAxes{};

      for (std::size_t i{}; i < shapeRank; ++i)
      {
        if (orgAxesOrder[i] >= shapeRank)
        {
          throw Exception{Error::invalidArgument, "axes order must not contain out-of-range values"};
        }

        if (seenAxes.test(orgAxesOrder[i]))
        {
          throw Exception{Error::invalidArgument, "axes order must not contain duplicates"};
        }

        seenAxes.set(orgAxesOrder[i]);
      }
    }

    strides[orgAxesOrder[shapeRank - 1]] = fastestAxisStride;

    for (std::size_t i = shapeRank - 1; i > 0; --i)
    {
      strides[orgAxesOrder[i - 1]] = detail::safeIntCast<StrideT>(shape[i]) * strides[orgAxesOrder[i]];
    }
  }
} // namespace afft

#endif /* AFFT_MEMORY_HPP */
