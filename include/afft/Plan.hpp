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
# include "detail/include.hpp"
#endif

#include "backend.hpp"
#include "common.hpp"
#include "memory.hpp"
#include "target.hpp"
#include "transform.hpp"
#include "detail/Desc.hpp"
#include "detail/validate.hpp"

AFFT_EXPORT namespace afft
{
  class Plan : public std::enable_shared_from_this<Plan>
  {
    friend struct detail::DescGetter; 

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

      /// @brief Get multi-process backend.
      [[nodiscard]] constexpr MpBackend getMpBackend() const
      {
        return mDesc.getMpBackend();
      }

      /**
       * @brief Get multi-process backend parameters.
       * @tparam mpBackend Multi-process backend type
       * @return Multi-process backend parameters
       */
      template<MpBackend mpBackend>
      [[nodiscard]] constexpr MpBackendParameters<mpBackend> getMpBackendParameters() const
      {
        static_assert(detail::isValid(mpBackend), "invalid multi-process backend type");

        if (mpBackend != getMpBackend())
        {
          throw std::invalid_argument("plan multi-process backend does not match requested multi-process backend");
        }

        return mDesc.getMpParameters<mpBackend>();
      }

      /// @brief Get the transform.
      [[nodiscard]] constexpr Transform getTransform() const
      {
        return mDesc.getTransform();
      }

      /**
       * @brief Get transform parameters.
       * @tparam transform Transform type
       * @return Transform parameters
       */
      template<Transform transform>
      [[nodiscard]] constexpr TransformParameters<transform> getTransformParameters() const
      {
        static_assert(detail::isValid(transform), "invalid transform type");

        if (transform != getTransform())
        {
          throw std::invalid_argument("plan transform does not match requested transform");
        }

        return mDesc.getTransformParameters<transform>();
      }

      /**
       * @brief Get target.
       * @return Target
       */
      [[nodiscard]] constexpr Target getTarget() const noexcept
      {
        return mDesc.getTarget();
      }

      /**
       * @brief Get target count.
       * @return Target count.
       */
      [[nodiscard]] constexpr std::size_t getTargetCount() const noexcept
      {
        return mDesc.getTargetCount();
      }

      /**
       * @brief Get target parameters.
       * @tparam target Target type
       * @return Target parameters
       */
      template<Target target>
      [[nodiscard]] constexpr TargetParameters<target> getTargetParameters() const
      {
        static_assert(detail::isValid(target), "invalid target type");

        if (target != getTarget())
        {
          throw std::invalid_argument("plan target does not match requested target");
        }

        return mDesc.getTargetParameters<target>();
      }

      /**
       * @brief Get memory layout.
       * @return Memory layout.
       */
      [[nodiscard]] MemoryLayout getMemoryLayout() const noexcept
      {
        return mDesc.getMemoryLayout();
      }

      /**
       * @brief Get backend.
       * @return Backend
       */
      [[nodiscard]] virtual Backend getBackend() const noexcept = 0;

      /**
       * @brief Get element count of the source buffers.
       * @return Element count of the source buffers.
       */
      [[nodiscard]] virtual View<std::size_t> getSrcElemCounts() const noexcept = 0;

      /**
       * @brief Get element count of the destination buffers.
       * @return Element count of the destination buffers.
       */
      [[nodiscard]] virtual View<std::size_t> getDstElemCounts() const noexcept = 0;

      /**
       * @brief Get workspace size.
       * @return Workspace size.
       */
      [[nodiscard]] virtual View<std::size_t> getWorkspaceSizes() const noexcept = 0;

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

        executeImpl1(makeScalarView(srcDst), makeScalarView(srcDst), execParams);
      }

      /**
       * @brief Execute the plan.
       * @tparam SrcDstT Source/destination type.
       * @tparam ExecParamsT Execution parameters type.
       * @param srcDst Source/destination buffers.
       * @param execParams Execution parameters.
       */
      template<typename SrcDstT, typename ExecParamsT = DefaultExecParams>
      void execute(View<SrcDstT*> srcDst, const ExecParamsT execParams = {})
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

        executeImpl1(makeScalarView(src), makeScalarView(dst), execParams);
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
      void execute(View<SrcT*> src, View<DstT*> dst, const ExecParamsT execParams = {})
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

        executeImpl1(makeScalarView(src), makeScalarView(dst), execParams);
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

        executeImpl1(makeScalarView(src), makeScalarView(dst), execParams);
      }

      /**
       * @brief Execute the plan without type checking.
       * @tparam ExecParamsT Execution parameters type.
       * @param src Source buffer.
       * @param dst Destination buffer.
       * @param execParams Execution parameters.
       */
      template<typename ExecParamsT = DefaultExecParams>
      void executeUnsafe(View<const void*> src, View<void*> dst, const ExecParamsT& execParams = {})
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
      void executeUnsafe(View<void*> src, View<void*> dst, const ExecParamsT& execParams = {})
      {
        static_assert(isKnownExecParams<ExecParamsT>, "invalid execution parameters type");

        executeImpl1(src, dst, execParams);
      }

#   ifdef AFFT_ENABLE_OPENCL
      void executeUnsafe(cl_mem srcDst, const afft::opencl::ExecutionParameters& execParams = {})
      {
        executeUnsafe(srcDst, srcDst, execParams);
      }

      void executeUnsafe(View<cl_mem> srcDst, const afft::opencl::ExecutionParameters& execParams = {})
      {
        executeUnsafe(srcDst, srcDst, execParams);
      }

      void executeUnsafe(cl_mem src, cl_mem dst, const afft::opencl::ExecutionParameters& execParams = {})
      {
        executeUnsafe(makeScalarView(src), makeScalarView(dst), execParams);
      }

      void executeUnsafe(View<cl_mem> src, View<cl_mem> dst, const afft::opencl::ExecutionParameters& execParams = {})
      {

      }
#   endif /* AFFT_ENABLE_OPENCL */
    protected:
      /// @brief Default constructor is deleted.
      Plan() = delete;

      /// @brief Constructor.
      Plan(const detail::Desc& desc)
      : mDesc{desc}
      {}

      /// @brief Get the plan description.
      [[nodiscard]] const detail::Desc& getDesc() const noexcept
      {
        return mDesc;
      }

      /**
       * @brief Execute the plan backend implementation.
       * @param src Source buffers.
       * @param dst Destination buffers.
       * @param execParams Execution parameters.
       */
      virtual void executeBackendImpl([[maybe_unused]] View<void*>                     src,
                                      [[maybe_unused]] View<void*>                     dst,
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
      virtual void executeBackendImpl([[maybe_unused]] View<void*>                            src,
                                      [[maybe_unused]] View<void*>                            dst,
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
      virtual void executeBackendImpl([[maybe_unused]] View<void*>                           src,
                                      [[maybe_unused]] View<void*>                           dst,
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
      virtual void executeBackendImpl([[maybe_unused]] View<void*>                              src,
                                      [[maybe_unused]] View<void*>                              dst,
                                      [[maybe_unused]] const afft::opencl::ExecutionParameters& execParams)
      {
        throw std::logic_error{"backend does not implement opencl execution"};
      }

    
      detail::Desc mDesc;
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
            throw std::invalid_argument{"invalid type precision"};
          }

          if (srcComplexity != refSrcCmpl && srcComplexity != refDstCmpl)
          {
            throw std::invalid_argument{"invalid type complexity"};
          }
          break;
        case Placement::outOfPlace:
          if (srcPrecision != prec.source)
          {
            throw std::invalid_argument{"invalid source type precision"};
          }
          
          if (dstPrecision != prec.destination)
          {
            throw std::invalid_argument{"invalid destination type precision"};
          }

          if (srcComplexity != refSrcCmpl)
          {
            throw std::invalid_argument{"invalid source type complexity"};
          }

          if (dstComplexity != refDstCmpl)
          {
            throw std::invalid_argument{"invalid destination type complexity"};
          }
          break;
        default:
          detail::cxx::unreachable();
        }
      }

      /// @brief Check if the source is preserved.
      void checkSrcIsPreserved() const
      {
        if (mDesc.isDestructive())
        {
          throw std::invalid_argument("running destructive transform on const source data");
        }
      }

      /**
       * @brief Check execution buffer count.
       * @param srcCount Source buffer count.
       * @param dstCount Destination buffer count.
       */
      void checkBufferCount(const std::size_t srcCount, const std::size_t dstCount) const
      {
        const auto targetCount = mDesc.getTargetCount();

        if (srcCount != targetCount)
        {
          throw std::invalid_argument{"invalid source buffer count"};
        }

        if (dstCount != targetCount)
        {
          throw std::invalid_argument{"invalid destination buffer count"};
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
          throw std::invalid_argument("placement does not match plan placement");
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
      void executeImpl1(View<SrcT*> src, View<DstT*> dst, const ExecParamsT& execParams)
      {
        static_assert((std::is_void_v<SrcT> && std::is_void_v<DstT>) ||
                      (!std::is_void_v<SrcT> && !std::is_void_v<DstT>), "invalid source and destination types");

        using NonConstSrcT = std::remove_const_t<SrcT>;

        if constexpr (std::is_const_v<SrcT>)
        {
          checkSrcIsPreserved();
        }

        View<NonConstSrcT*> srcNonConst(const_cast<NonConstSrcT* const*>(src.data()), src.size());

        checkBufferCount(src.size(), dst.size());
        
        const bool isInPlace = std::equal(src.begin(), src.end(), dst.begin(), [](const auto& s, const auto& d)
        {
          return reinterpret_cast<std::uintptr_t>(s) == reinterpret_cast<std::uintptr_t>(d);
        });
        checkPlacement((isInPlace) ? Placement::inPlace : Placement::outOfPlace);

        if constexpr (!std::is_void_v<SrcT> && !std::is_void_v<DstT>)
        {
          checkExecTypeProps(typePrecision<SrcT>, typeComplexity<SrcT>, typePrecision<DstT>, typeComplexity<DstT>);
        }

        executeImpl2(srcNonConst, dst, execParams);
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
      void executeImpl2(View<SrcT> src, View<DstT> dst, const ExecParamsT& execParams)
      {
        auto isNullPtr = [](const auto* ptr) { return ptr == nullptr; };

        if (std::any_of(src.begin(), src.end(), isNullPtr))
        {
          throw std::invalid_argument("a null pointer was passed as source buffer");
        }

        if (std::any_of(dst.begin(), dst.end(), isNullPtr))
        {
          throw std::invalid_argument("a null pointer was passed as destination buffer");
        }

        View<void*> srcVoid{reinterpret_cast<void* const*>(src.data()), src.size()};
        View<void*> dstVoid{reinterpret_cast<void* const*>(dst.data()), dst.size()};

        if constexpr (std::is_same_v<ExecParamsT, DefaultExecParams>)
        {
          switch (getTarget())
          {
          case Target::cpu:
            executeBackendImpl(srcVoid, dstVoid, afft::cpu::ExecutionParameters{});
            break;
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
            throw std::invalid_argument("execution parameters target does not match plan target");
          }

          executeBackendImpl(srcVoid, dstVoid, execParams);
        }
      }
  };
} // namespace afft

#endif /* AFFT_PLAN_HPP */
