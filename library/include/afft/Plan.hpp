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

#ifndef AFFT_PLAN_HPP
#define AFFT_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include <afft/detail/include.hpp>
#endif

#include <afft/backend.hpp>
#include <afft/Description.hpp>
#include <afft/common.hpp>
#include <afft/memory.hpp>
#include <afft/target.hpp>
#include <afft/transform.hpp>
#include <afft/detail/Desc.hpp>
#include <afft/detail/validate.hpp>

AFFT_EXPORT namespace afft
{
  /// @brief Plan base class.
  class Plan : public std::enable_shared_from_this<Plan>, protected Description
  {
    private:
      /// @brief Default execution parameters helper.
      struct DefaultExecParams
      {
        /// @brief Conversion to any execution parameters.
        template<typename ExecParamsT>
        [[nodiscard]] constexpr operator ExecParamsT() const noexcept
        {
          static_assert(isExecutionParameters<ExecParamsT>, "invalid execution parameters type");

          return ExecParamsT{};
        }
      };

      /**
       * @brief Check if the type is known execution parameters.
       * @tparam T Type to check.
       */
      template<typename T>
      static constexpr bool isKnownExecParams = isExecutionParameters<T> || std::is_same_v<T, DefaultExecParams>;

    public:
      /// @brief Copy constructor is deleted.
      Plan(const Plan&) = delete;

      /// @brief Move constructor.
      Plan(Plan&&) = default;

      /// @brief Destructor.
      virtual ~Plan() = default;

      /// @brief Copy assignment operator is deleted.
      Plan& operator=(const Plan&) = delete;

      /// @brief Move assignment operator.
      Plan& operator=(Plan&&) = default;

      /**
       * @brief Get the plan description.
       * @return Plan description.
       */
      [[nodiscard]] constexpr const Description& getDescription() const noexcept
      {
        return *this;
      }

      /**
       * @brief Get backend.
       * @return Backend
       */
      [[nodiscard]] virtual Backend getBackend() const noexcept = 0;

      /**
       * @brief Get the backend parameters.
       * @tparam mpBackend Multi-process backend.
       * @tparam target Target.
       * @return Backend parameters.
       */
      template<MpBackend mpBackend, Target target>
      [[nodiscard]] constexpr const BackendParameters<mpBackend, target>& getBackendParameters() const noexcept
      {
        if (mpBackend != mDesc.getMpBackend())
        {
          throw Exception{Error::invalidArgument, "mpBackend does not match the actual multi-process backend"};
        }

        if (target != mDesc.getTarget())
        {
          throw Exception{Error::invalidArgument, "target does not match the actual target"};
        }

        return *static_cast<const BackendParameters<mpBackend, target>*>(getBackendParametersImpl());
      }

      /**
       * @brief Get the backend parameters variant.
       * @return Backend parameters variant.
       */
      [[nodiscard]] virtual BackendParametersVariant getBackendParametersVariant() const noexcept = 0;

      /**
       * @brief Does the plan overwrite the source buffer?
       * @return True if the plan is destructive.
       */
      [[nodiscard]] constexpr bool isDestructive() const noexcept
      {
        return mIsDestructive;
      }

      /**
       * @brief Get the precision of the source and destination buffers.
       * @return Source and destination buffer precision.
       */
      [[nodiscard]] constexpr std::pair<Precision, Precision> getSrcDstPrecision() const noexcept
      {
        const auto& prec = mDesc.getPrecision();
        return std::make_pair(prec.source, prec.destination);
      }

      /**
       * @brief Get source and destination buffer complexity.
       * @return Source and destination buffer complexity.
       */
      [[nodiscard]] constexpr std::pair<Complexity, Complexity> getSrcDstComplexity() const noexcept
      {
        return mDesc.getSrcDstComplexity();
      }

      /**
       * @brief Get element count of the source buffers. If workspace is enlargedBuffer, the element count may be larger
       *        than the transform size.
       * @return Element count of the source buffers.
       */
      [[nodiscard]] virtual const std::size_t* getSrcElemCounts() const noexcept = 0;

      /**
       * @brief Get element count of the destination buffers. If workspace is enlargedBuffer, the element count may be
       *        larger than the transform size.
       * @return Element count of the destination buffers.
       */
      [[nodiscard]] virtual const std::size_t* getDstElemCounts() const noexcept = 0;

      /**
       * @brief Get external workspace sizes. Only valid if the workspace is external.
       * @return External workspace sizes.
       */
      [[nodiscard]] virtual const std::size_t* getExternalWorkspaceSizes() const noexcept
      {
        return {};
      }

      /**
       * @brief Execute the plan.
       * @tparam SrcDstT Source/destination type.
       * @tparam ExecParamsT Execution parameters type.
       * @param srcDst Source/destination buffer.
       * @param execParams Execution parameters.
       */
      template<typename SrcDstT, typename ExecParamsT = DefaultExecParams>
      void execute(SrcDstT* srcDst, const ExecParamsT execParams = {})
      {
        static_assert(isKnownType<SrcDstT>, "unknown source/destination type");
        static_assert(!std::is_const_v<SrcDstT>, "source/destination type must be non-const");
        static_assert(isKnownExecParams<ExecParamsT>, "invalid execution parameters type");

        executeImpl1(std::addressof(srcDst), std::addressof(srcDst), execParams);
      }

      /**
       * @brief Execute the plan.
       * @tparam SrcDstT Source/destination type.
       * @tparam ExecParamsT Execution parameters type.
       * @param srcDst Source/destination buffers.
       * @param execParams Execution parameters.
       */
      template<typename SrcDstT, typename ExecParamsT = DefaultExecParams>
      void execute(SrcDstT* const* srcDst, const ExecParamsT execParams = {})
      {
        static_assert(isKnownType<SrcDstT>, "unknown source/destination type");
        static_assert(!std::is_const_v<SrcDstT>, "source/destination type must be non-const");
        static_assert(isKnownExecParams<ExecParamsT>, "invalid execution parameters type");

        executeImpl1(srcDst, srcDst, execParams);
      }

      /**
       * @brief Execute the plan.
       * @tparam SrcT Source type.
       * @tparam DstT Destination type.
       * @tparam ExecParamsT Execution parameters type.
       * @param src Source buffer.
       * @param dst Destination buffer.
       * @param execParams Execution parameters.
       */
      template<typename SrcT, typename DstT, typename ExecParamsT = DefaultExecParams>
      void execute(SrcT* src, DstT* dst, const ExecParamsT execParams = {})
      {
        static_assert(isKnownType<SrcT>, "unknown source type");
        static_assert(isKnownType<DstT>, "unknown destination type");
        static_assert(!std::is_const_v<DstT>, "destination type must be non-const");
        static_assert(isKnownExecParams<ExecParamsT>, "invalid execution parameters type");

        executeImpl1(std::addressof(src), std::addressof(dst), execParams);
      }

      /**
       * @brief Execute the plan.
       * @tparam SrcT Source type.
       * @tparam DstT Destination type.
       * @tparam ExecParamsT Execution parameters type.
       * @param src Source buffers.
       * @param dst Destination buffers.
       * @param execParams Execution parameters.
       */
      template<typename SrcT, typename DstT, typename ExecParamsT = DefaultExecParams>
      void execute(SrcT* const* src, DstT* const* dst, const ExecParamsT execParams = {})
      {
        static_assert(isKnownType<SrcT>, "unknown source type");
        static_assert(isKnownType<DstT>, "unknown destination type");
        static_assert(!std::is_const_v<DstT>, "destination type must be non-const");
        static_assert(isKnownExecParams<ExecParamsT>, "invalid execution parameters type");

        executeImpl1(src, dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking.
       * @tparam ExecParamsT Execution parameters type.
       * @param src Source buffer.
       * @param dst Destination buffer.
       * @param execParams Execution parameters.
       */
      template<typename ExecParamsT = DefaultExecParams>
      void executeUnsafe(const void* src, void* dst, const ExecParamsT& execParams = {})
      {
        static_assert(isKnownExecParams<ExecParamsT>, "invalid execution parameters type");

        executeImpl1(std::addressof(src), std::addressof(dst), execParams);
      }

      /**
       * @brief Execute the plan without type checking.
       * @tparam ExecParamsT Execution parameters type.
       * @param src Source buffer.
       * @param dst Destination buffer.
       * @param execParams Execution parameters.
       */
      template<typename ExecParamsT = DefaultExecParams>
      void executeUnsafe(void* src, void* dst, const ExecParamsT& execParams = {})
      {
        static_assert(isKnownExecParams<ExecParamsT>, "invalid execution parameters type");

        executeImpl1(std::addressof(src), std::addressof(dst), execParams);
      }

      /**
       * @brief Execute the plan without type checking.
       * @tparam ExecParamsT Execution parameters type.
       * @param src Source buffer.
       * @param dst Destination buffer.
       * @param execParams Execution parameters.
       */
      template<typename ExecParamsT = DefaultExecParams>
      void executeUnsafe(const void* const* src, void* const* dst, const ExecParamsT& execParams = {})
      {
        static_assert(isKnownExecParams<ExecParamsT>, "invalid execution parameters type");

        executeImpl1(src, dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking.
       * @tparam ExecParamsT Execution parameters type.
       * @param src Source buffer.
       * @param dst Destination buffer.
       * @param execParams Execution parameters.
       */
      template<typename ExecParamsT = DefaultExecParams>
      void executeUnsafe(void* const* src, void* const* dst, const ExecParamsT& execParams = {})
      {
        static_assert(isKnownExecParams<ExecParamsT>, "invalid execution parameters type");

        executeImpl1(src, dst, execParams);
      }

#   ifdef AFFT_ENABLE_OPENCL
      void executeUnsafe(cl_mem srcDst, const afft::opencl::ExecutionParameters& execParams = {})
      {
        executeUnsafe(srcDst, srcDst, execParams);
      }

      void executeUnsafe(const cl_mem* srcDst, const afft::opencl::ExecutionParameters& execParams = {})
      {
        executeUnsafe(srcDst, srcDst, execParams);
      }

      void executeUnsafe(cl_mem src, cl_mem dst, const afft::opencl::ExecutionParameters& execParams = {})
      {
        executeUnsafe(std::addressof(src), std::addressof(dst), execParams);
      }

      void executeUnsafe(const cl_mem* src, const cl_mem* dst, const afft::opencl::ExecutionParameters& execParams = {})
      {

      }
#   endif /* AFFT_ENABLE_OPENCL */
    protected:
      /// @brief Default constructor is deleted.
      Plan() = delete;

      /// @brief Constructor.
      Plan(const Description& desc)
      : Description{desc}
      {}

      /**
       * @brief Get the backend parameters implementation.
       * @return Backend parameters implementation.
       */
      [[nodiscard]] virtual const void* getBackendParametersImpl() const noexcept = 0;

      /**
       * @brief Execute the plan backend implementation.
       * @param src Source buffers.
       * @param dst Destination buffers.
       * @param execParams Execution parameters.
       */
      virtual void executeBackendImpl([[maybe_unused]] void* const*                    src,
                                      [[maybe_unused]] void* const*                    dst,
                                      [[maybe_unused]] const cpu::ExecutionParameters& execParams)
      {
        throw std::logic_error{"backend does not implement cpu execution"};
      }

      /**
       * @brief Execute the plan backend implementation.
       * @param src Source buffers.
       * @param dst Destination buffers.
       * @param execParams Execution parameters.
       */
      virtual void executeBackendImpl([[maybe_unused]] void* const*                           src,
                                      [[maybe_unused]] void* const*                           dst,
                                      [[maybe_unused]] const afft::cuda::ExecutionParameters& execParams)
      {
        throw std::logic_error{"backend does not implement cuda execution"};
      }

      /**
       * @brief Execute the plan backend implementation.
       * @param src Source buffers.
       * @param dst Destination buffers.
       * @param execParams Execution parameters.
       */
      virtual void executeBackendImpl([[maybe_unused]] void* const*                          src,
                                      [[maybe_unused]] void* const*                          dst,
                                      [[maybe_unused]] const afft::hip::ExecutionParameters& execParams)
      {
        throw std::logic_error{"backend does not implement hip execution"};
      }

      /**
       * @brief Execute the plan backend implementation.
       * @param src Source buffers.
       * @param dst Destination buffers.
       * @param execParams Execution parameters.
       */
      virtual void executeBackendImpl([[maybe_unused]] void* const*                             src,
                                      [[maybe_unused]] void* const*                             dst,
                                      [[maybe_unused]] const afft::opencl::ExecutionParameters& execParams)
      {
        throw std::logic_error{"backend does not implement opencl execution"};
      }

      /**
       * @brief Execute the plan backend implementation.
       * @param src Source buffers.
       * @param dst Destination buffers.
       * @param execParams Execution parameters.
       */
      virtual void executeBackendImpl([[maybe_unused]] void* const*                             src,
                                      [[maybe_unused]] void* const*                             dst,
                                      [[maybe_unused]] const afft::openmp::ExecutionParameters& execParams)
      {
        throw std::logic_error{"backend does not implement openmp execution"};
      }

      bool mIsDestructive{mDesc.getPlacement() == Placement::inPlace}; ///< Is the plan destructive?
    private:
      /**
       * @brief Check execution type properties.
       * @param srcPrecision Source precision.
       * @param srcComplexity Source complexity.
       * @param dstPrecision Destination precision.
       * @param dstComplexity Destination complexity.
       */
      void checkExecTypeProps(const Precision  srcPrecision,
                              const Complexity srcComplexity,
                              const Precision  dstPrecision,
                              const Complexity dstComplexity) const
      {
        const auto& prec = mDesc.getPrecision();
        const auto [refSrcCmpl, refDstCmpl] = mDesc.getSrcDstComplexity();

        switch (mDesc.getPlacement())
        {
        case Placement::inPlace:
          if (srcPrecision != prec.source && srcPrecision != prec.destination)
          {
            throw Exception{Error::invalidArgument, "invalid type precision"};
          }

          if (srcComplexity != refSrcCmpl && srcComplexity != refDstCmpl)
          {
            throw Exception{Error::invalidArgument, "invalid type complexity"};
          }
          break;
        case Placement::outOfPlace:
          if (srcPrecision != prec.source)
          {
            throw Exception{Error::invalidArgument, "invalid source type precision"};
          }
          
          if (dstPrecision != prec.destination)
          {
            throw Exception{Error::invalidArgument, "invalid destination type precision"};
          }

          if (srcComplexity != refSrcCmpl)
          {
            throw Exception{Error::invalidArgument, "invalid source type complexity"};
          }

          if (dstComplexity != refDstCmpl)
          {
            throw Exception{Error::invalidArgument, "invalid destination type complexity"};
          }
          break;
        default:
          detail::cxx::unreachable();
        }
      }

      /// @brief Check if the source is preserved.
      void checkSrcIsPreserved() const
      {
        if (isDestructive())
        {
          throw Exception{Error::invalidArgument, "running destructive transform on const source data"};
        }
      }

      /**
       * @brief Check placement.
       * @param placement Placement.
       */
      void checkPlacement(const Placement placement) const
      {
        if (placement != mDesc.getPlacement())
        {
          throw Exception{Error::invalidArgument, "placement does not match plan placement"};
        }
      }

      /**
       * @brief Level 1 implementation of the execute method.
       * @tparam SrcT Source type.
       * @tparam DstT Destination type.
       * @tparam ExecParamsT Execution parameters type.
       * @param src Source buffers.
       * @param dst Destination buffers.
       * @param execParams Execution parameters.
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void executeImpl1(SrcT* const* src, DstT* const* dst, const ExecParamsT& execParams)
      {
        static_assert((std::is_void_v<SrcT> && std::is_void_v<DstT>) ||
                      (!std::is_void_v<SrcT> && !std::is_void_v<DstT>), "invalid source and destination types");

        if constexpr (std::is_const_v<SrcT>)
        {
          checkSrcIsPreserved();
        }

        const auto [srcBufferCount, dstBufferCount] = mDesc.getSrcDstBufferCount();
        
        const bool isInPlace = std::equal(src,
                                          src + srcBufferCount,
                                          dst,
                                          dst + dstBufferCount,
                                          [](const auto& s, const auto& d)
        {
          return reinterpret_cast<detail::cxx::uintptr_t>(s) == reinterpret_cast<detail::cxx::uintptr_t>(d);
        });

        checkPlacement((isInPlace) ? Placement::inPlace : Placement::outOfPlace);

        if constexpr (!std::is_void_v<SrcT> && !std::is_void_v<DstT>)
        {
          checkExecTypeProps(precisionOf<SrcT>, complexityOf<SrcT>, precisionOf<DstT>, complexityOf<DstT>);
        }

        executeImpl2(const_cast<std::remove_const_t<SrcT>* const*>(src), dst, execParams);
      }

      /**
       * @brief Level 2 implementation of the execute method.
       * @tparam SrcT Source type.
       * @tparam DstT Destination type.
       * @tparam ExecParamsT Execution parameters type.
       * @param src Source buffers.
       * @param dst Destination buffers.
       * @param execParams Execution parameters.
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void executeImpl2(SrcT* const* src, DstT* const* dst, const ExecParamsT& execParams)
      {
        auto isNullPtr = [](const auto* ptr) { return ptr == nullptr; };

        const auto [srcBufferCount, dstBufferCount] = mDesc.getSrcDstBufferCount();

        if (std::any_of(src, src + srcBufferCount, isNullPtr))
        {
          throw Exception{Error::invalidArgument, "a null pointer was passed as source buffer"};
        }

        if (std::any_of(dst, dst + dstBufferCount, isNullPtr))
        {
          throw Exception{Error::invalidArgument, "a null pointer was passed as destination buffer"};
        }

        void* const* srcVoid = reinterpret_cast<void* const*>(src);
        void* const* dstVoid = reinterpret_cast<void* const*>(dst);

        if constexpr (std::is_same_v<ExecParamsT, DefaultExecParams>)
        {
          switch (getTarget())
          {
#         ifdef AFFT_ENABLE_CPU
          case Target::cpu:
            executeBackendImpl(srcVoid, dstVoid, afft::cpu::ExecutionParameters{});
            break;
#         endif
#         ifdef AFFT_ENABLE_CUDA
          case Target::cuda:
            executeBackendImpl(srcVoid, dstVoid, afft::cuda::ExecutionParameters{});
            break;
#         endif
#         ifdef AFFT_ENABLE_HIP
          case Target::hip:
            executeBackendImpl(srcVoid, dstVoid, afft::hip::ExecutionParameters{});
            break;
#         endif
#         ifdef AFFT_ENABLE_OPENCL
          case Target::opencl:
            executeBackendImpl(srcVoid, dstVoid, afft::opencl::ExecutionParameters{});
            break;
#         endif
          default:
            throw std::logic_error{"disabled target"};
          }
        }
        else
        {
          if (execParams.target != getTarget())
          {
            throw Exception{Error::invalidArgument, "execution parameters target does not match plan target"};
          }

          executeBackendImpl(srcVoid, dstVoid, execParams);
        }
      }
  };
} // namespace afft

#endif /* AFFT_PLAN_HPP */
