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

namespace afft
{
  /// @brief Alignment of a data type
  enum class Alignment : std::size_t
  {
    defaultNew = __STDCPP_DEFAULT_NEW_ALIGNMENT__, ///< Default alignment for new operator
    simd128    = 16,                               ///< 128-bit SIMD alignment
    simd256    = 32,                               ///< 256-bit SIMD alignment
    simd512    = 64,                               ///< 512-bit SIMD alignment
    simd1024   = 128,                              ///< 1024-bit SIMD alignment
    simd2048   = 256,                              ///< 2048-bit SIMD alignment

    sse    = simd128,  ///< SSE alignment
    sse2   = simd128,  ///< SSE2 alignment
    sse3   = simd128,  ///< SSE3 alignment
    sse4   = simd128,  ///< SSE4 alignment
    sse4_1 = simd128,  ///< SSE4.1 alignment
    sse4_2 = simd128,  ///< SSE4.2 alignment
    avx    = simd256,  ///< AVX alignment
    avx2   = simd256,  ///< AVX2 alignment
    avx512 = simd512,  ///< AVX-512 alignment
    neon   = simd128,  ///< NEON alignment
    sve    = simd2048, ///< SVE alignment
  };
  
  /// @brief Memory layout of the centralized transform
  struct MemoryLayout
  {
    ComplexFormat complexFormat{ComplexFormat::interleaved}; ///< complex number format
    Alignment     alignment{};                               ///< alignment of the memory
    View<Size>    srcStrides{};                              ///< strides of the source memory
    View<Size>    dstStrides{};                              ///< strides of the destination memory
  };

  /// @brief Memory block
  struct MemoryBlock
  {
    View<Size> starts{};    ///< starts of the memory block
    View<Size> sizes{};     ///< sizes of the memory block
    View<Size> strides{};   ///< strides of the memory block
  };

  /// @brief Memory layout of the distributed transform
  struct DistribMemoryLayout
  {
    ComplexFormat     complexFormat{ComplexFormat::interleaved}; ///< complex number format
    Alignment         alignment{};                               ///< alignment of the memory
    View<MemoryBlock> srcBlocks{};                               ///< source memory blocks
    View<Axis>        srcDistribAxes{};                          ///< axes along which the source data are distributed
    View<Axis>        srcAxesOrder{};                            ///< order of the source axes
    View<MemoryBlock> dstBlocks{};                               ///< destination memory blocks
    View<Axis>        dstDistribAxes{};                          ///< axes along which the destination data are distributed
    View<Axis>        dstAxesOrder{};                            ///< order of the destination axes
  };

namespace cpu
{
  /// @brief Default alignment for memory allocation
#if defined(__AVX512F__)
  inline constexpr auto defaultAlignment = Alignment::avx512;
#elif defined(__AVX2__)
  inline constexpr auto defaultAlignment = Alignment::avx2;
#elif defined(__AVX__)
  inline constexpr auto defaultAlignment = Alignment::avx;
#elif defined(__SSE4_2__)
  inline constexpr auto defaultAlignment = Alignment::sse4_2;
#elif defined(__SSE4_1__)
  inline constexpr auto defaultAlignment = Alignment::sse4_1;
#elif defined(__SSE4__)
  inline constexpr auto defaultAlignment = Alignment::sse4;
#elif defined(__SSE3__)
  inline constexpr auto defaultAlignment = Alignment::sse3;
#elif defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP == 2)
  inline constexpr auto defaultAlignment = Alignment::sse2;
#elif defined(__SSE__) || (defined(_M_IX86_FP) && _M_IX86_FP == 1)
  inline constexpr auto defaultAlignment = Alignment::sse;
#elif defined(__ARM_NEON) || defined(_M_ARM_NEON)
  inline constexpr auto defaultAlignment = Alignment::neon;
#elif (defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE == 1) || (defined(__ARM_FEATURE_SVE2) && __ARM_FEATURE_SVE2 == 1)
  inline constexpr auto defaultAlignment = Alignment::sve;
#else
  inline constexpr auto defaultAlignment = Alignment::defaultNew;
#endif

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
} // namespace cpu

  namespace cuda
  {
    /**
     * @class ManagedAllocator
     * @brief Allocator named concept implementation implementation for unified cuda memory to be used with std::vector and
     *        others.
     * @tparam T Type of the memory
     */
    template<typename T>
    class ManagedAllocator;
  } // namespace cuda

#ifdef AFFT_ENABLE_CUDA
  /**
   * @class ManagedAllocator
   * @brief Allocator named concept implementation implementation for unified cuda memory to be used with std::vector and
   *        others.
   * @tparam T Type of the memory
   */
  template<typename T>
  class cuda::ManagedAllocator
  {
    public:
      /// @brief Type of the memory
      using value_type = T;
      
      /// @brief Default constructor
      constexpr ManagedAllocator(unsigned flags = cudaMemAttachGlobal) noexcept
      : mFlags{flags}
      {}

      /// @brief Copy constructor
      template<typename U>
      constexpr ManagedAllocator(const ManagedAllocator<U>& other) noexcept
      : mFlags{other.mFlags}
      {}

      /// @brief Move constructor
      template<typename U>
      constexpr ManagedAllocator(ManagedAllocator<U>&& other) noexcept
      : mFlags{std::move(other.mFlags)}
      {}

      /// @brief Destructor
      ~ManagedAllocator() noexcept = default;

      /// @brief Copy assignment operator
      template<typename U>
      constexpr ManagedAllocator& operator=(const ManagedAllocator<U>& other) noexcept
      {
        mFlags = other.mFlags;
        return *this;
      }

      /// @brief Move assignment operator
      template<typename U>
      constexpr ManagedAllocator& operator=(ManagedAllocator<U>&& other) noexcept
      {
        mFlags = std::move(other.mFlags);
        return *this;
      }

      /**
       * @brief Allocate memory
       * @param n Number of elements
       * @return Pointer to the allocated memory
       */
      [[nodiscard]] T* allocate(std::size_t n)
      {
        T* ptr{};

        detail::cuda::checkError(cudaMallocManaged(&ptr, n * sizeof(T), mFlags));

        return ptr;
      }

      /**
       * @brief Deallocate memory
       * @param p Pointer to the memory
       * @param n Number of elements
       */
      void deallocate(T* p, std::size_t) noexcept
      {
        detail::cuda::checkError(cudaFree(p));
      }

      /**
       * @brief Get flags
       * @return Flags
       */
      [[nodiscard]] constexpr unsigned getFlags() const noexcept
      {
        return mFlags;
      }
    private:
      unsigned mFlags{}; ///< Flags for memory allocation
  };
#endif

  namespace hip
  {
    /**
     * @class ManagedAllocator
     * @brief Allocator named concept implementation implementation for unified hip memory to be used with std::vector and
     *        others.
     * @tparam T Type of the memory
     */
    template<typename T>
    class ManagedAllocator;
  } // namespace hip

#ifdef AFFT_ENABLE_HIP
  /**
   * @class ManagedAllocator
   * @brief Allocator named concept implementation implementation for unified hip memory to be used with std::vector and
   *        others.
   * @tparam T Type of the memory
   */
  template<typename T>
  class hip::ManagedAllocator
  {
    public:
      /// @brief Type of the memory
      using value_type = T;
      
      /// @brief Default constructor
      constexpr ManagedAllocator(unsigned flags = hipMemAttachGlobal) noexcept
      : mFlags{flags}
      {}

      /// @brief Copy constructor
      template<typename U>
      constexpr ManagedAllocator(const ManagedAllocator<U>& other) noexcept
      : mFlags{other.mFlags}
      {}

      /// @brief Move constructor
      template<typename U>
      constexpr ManagedAllocator(ManagedAllocator<U>&& other) noexcept
      : mFlags{std::move(other.mFlags)}
      {}

      /// @brief Destructor
      ~ManagedAllocator() noexcept = default;

      /// @brief Copy assignment operator
      template<typename U>
      constexpr ManagedAllocator& operator=(const ManagedAllocator<U>& other) noexcept
      {
        mFlags = other.mFlags;
        return *this;
      }

      /// @brief Move assignment operator
      template<typename U>
      constexpr ManagedAllocator& operator=(ManagedAllocator<U>&& other) noexcept
      {
        mFlags = std::move(other.mFlags);
        return *this;
      }

      /**
       * @brief Allocate memory
       * @param n Number of elements
       * @return Pointer to the allocated memory
       */
      [[nodiscard]] T* allocate(std::size_t n)
      {
        T* ptr{};

        detail::hip::checkError(hipMallocManaged(&ptr, n * sizeof(T)));

        return ptr;
      }

      /**
       * @brief Deallocate memory
       * @param p Pointer to the memory
       * @param n Number of elements
       */
      void deallocate(T* p, std::size_t) noexcept
      {
        detail::hip::checkError(hipFree(p));
      }

      /**
       * @brief Get flags
       * @return Flags
       */
      [[nodiscard]] constexpr unsigned getFlags() const noexcept
      {
        return mFlags;
      }
    private:
      unsigned mFlags{}; ///< Flags for memory allocation
  };
#endif

  namespace opencl
  {
    /**
     * @class SvmAllocator
     * @brief Allocator named concept implementation implementation for opencl shared virtual memory to be used with std::vector and
     *        others.
     * @tparam T Type of the memory
     */
    template<typename T>
    class SvmAllocator;
  } // namespace opencl

#if defined(AFFT_ENABLE_HIP) && defined(CL_VERSION_2_0)
  /**
   * @class SvmAllocator
   * @brief Allocator named concept implementation implementation for opencl shared virtual memory to be used with std::vector and
   *        others.
   * @tparam T Type of the memory
   */
  template<typename T>
  class opencl::SvmAllocator
  {
    public:
      /// @brief Type of the memory
      using value_type = T;
      
      /// @brief Default constructor not allowed
      SvmAllocator() = delete;

      /// @brief Default constructor not allowed
      SvmAllocator(cl_context context, cl_svm_mem_flags flags = CL_MEM_READ_WRITE, Alignment alignment = {})
      : mFlags{flags},
        mAlignment{alignment}
      {
        detail::opencl::checkError(clRetainContext(context));
        mContext.reset(context);
      }

      /// @brief Copy constructor
      template<typename U>
      SvmAllocator(const SvmAllocator<U>& other)
      : mFlags{other.getFlags()},
        mAlignment{other.getAlignment()}
      {
        detail::opencl::checkError(clRetainContext(other.getContext()));
        mContext.reset(other.getContext());
      }

      /// @brief Move constructor
      template<typename U>
      SvmAllocator(SvmAllocator<U>&& other) noexcept
      : mContext{std::move(other.mContext)},
        mFlags{std::move(other.mFlags)},
        mAlignment{std::move(other.mAlignment)}
      {}

      /// @brief Destructor
      ~SvmAllocator() = default;

      /// @brief Copy assignment operator
      template<typename U>
      SvmAllocator& operator=(const SvmAllocator<U>& other)
      {
        if (this != std::addressof(other))
        {
          detail::opencl::checkError(clRetainContext(other.getContext()));
          mContext.reset(other.getContext());

          mFlags     = other.getFlags();
          mAlignment = other.getAlignment();
        }

        return *this;
      }

      /// @brief Move assignment operator
      template<typename U>
      constexpr SvmAllocator& operator=(SvmAllocator<U>&& other) noexcept
      {
        if (this != std::addressof(other))
        {
          mContext   = std::move(other.mContext);
          mFlags     = std::move(other.mFlags);
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
        T* const ptr = static_cast<T*>(clSVMAlloc(mContext.get(),
                                                  mFlags,
                                                  n * sizeof(T),
                                                  static_cast<cl_uint>(mAlignment)));

        if (ptr == nullptr)
        {
          throw std::bad_alloc();
        }

        return ptr;
      }

      /**
       * @brief Deallocate memory
       * @param p Pointer to the memory
       * @param n Number of elements
       */
      void deallocate(T* p, std::size_t) noexcept
      {
        clSVMFree(mContext.get(), p);
      }

      /// @brief Get the OpenCL context
      [[nodiscard]] cl_context getContext() const noexcept
      {
        return mContext.get();
      }

      /// @brief Get the flags
      [[nodiscard]] cl_svm_mem_flags getFlags() const noexcept
      {
        return mFlags;
      }

      /// @brief Get the alignment
      [[nodiscard]] Alignment getAlignment() const noexcept
      {
        return mAlignment;
      }
    private:
      std::unique_ptr<std::remove_pointer_t<cl_context>, ContextDeleter> mContext;     ///< OpenCL context
      cl_svm_mem_flags                                                   mFlags{};     ///< Flags for memory allocation
      Alignment                                                          mAlignment{}; ///< Alignment for memory allocation
  };
#endif
} // namespace afft

#endif /* AFFT_MEMORY_HPP */
