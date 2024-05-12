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

#ifndef AFFT_DETAIL_PLAN_IMPL_HPP
#define AFFT_DETAIL_PLAN_IMPL_HPP

#include <array>
#include <cstddef>
#include <variant>

#include "Config.hpp"
#include "cxx.hpp"

namespace afft::detail
{
  class ExecParam
  {
    public:
      constexpr ExecParam() = default;
      constexpr ExecParam(void* realOrRealImag, void* imag = nullptr) noexcept
      {
        mBuffers[0] = realOrRealImag;
        mBuffers[1] = imag;
      }
      template<typename T>
      constexpr ExecParam(PlanarComplex<T> planarComplex) noexcept
      {
        mBuffers[0] = planarComplex.real;
        mBuffers[1] = planarComplex.imag;
      }
      constexpr ExecParam(const ExecParam&) = default;
      constexpr ExecParam(ExecParam&&) = default;
      ~ExecParam() = default;

      constexpr ExecParam& operator=(const ExecParam&) = default;
      constexpr ExecParam& operator=(ExecParam&&) = default;

      [[nodiscard]] constexpr bool isSplit() const noexcept
      {
        return mBuffers[1] != nullptr;
      }

      [[nodiscard]] constexpr void* getReal() const noexcept
      {
        return mBuffers[0];
      }

      template<typename T>
      [[nodiscard]] constexpr T* getRealAs() const noexcept
      {
        return static_cast<T*>(getReal());
      }

      [[nodiscard]] constexpr void* getRealImag() const noexcept
      {
        return mBuffers[0];
      }

      template<typename T>
      [[nodiscard]] constexpr T* getRealImagAs() const noexcept
      {
        return static_cast<T*>(getRealImag());
      }

      [[nodiscard]] constexpr void* getImag() const noexcept
      {
        return mBuffers[1];
      }

      template<typename T>
      [[nodiscard]] constexpr T* getImagAs() const noexcept
      {
        return static_cast<T*>(getImag());
      }

      [[nodiscard]] constexpr void** data() noexcept
      {
        return mBuffers.data();
      }

      template<typename T>
      [[nodiscard]] constexpr T** dataAs() noexcept
      {
        return reinterpret_cast<T**>(mBuffers.data());
      }
    protected:
    private:
      std::array<void*, 2> mBuffers{};
  };

  /**
   * @class PlanImpl
   * @brief The abstract base class for all plan implementations
   */
  class PlanImpl
  {
    public:
      /// @brief The default constructor is deleted
      PlanImpl() = delete;

      /// @brief The copy constructor is deleted
      PlanImpl(const PlanImpl&) = delete;

      /// @brief The move constructor is defaulted
      PlanImpl(PlanImpl&&) = default;

      /// @brief The destructor is defaulted
      virtual ~PlanImpl() = default;

      /// @brief The copy assignment operator is deleted
      PlanImpl& operator=(const PlanImpl&) = delete;

      /// @brief The move assignment operator is defaulted
      PlanImpl& operator=(PlanImpl&&) = default;

      /**
       * @brief Get the configuration of the plan
       * @return const reference to the configuration of the plan
       */
      [[nodiscard]] constexpr const Config& getConfig() const noexcept
      {
        return mConfig;
      }

      /**
       * @brief Execute the plan with default target parameters
       * @tparam ArgsT Argument types
       * @param args Arguments
       */
      template<typename... Args>
      void executeWithDefaultTargetParameters(Args&&... buffers)
      {
        switch (getConfig().getTarget())
        {
        case Target::cpu:
          execute(std::forward<Args>(buffers)..., afft::cpu::ExecutionParameters{});
          break;
        case Target::gpu:
          execute(std::forward<Args>(buffers)..., afft::gpu::ExecutionParameters{});
          break;
        default:
          cxx::unreachable();
        }
      }

      /**
       * @brief Execute in-place plan
       * @tparam SrcDstT Source or destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param srcDst Source and destination buffer
       * @param execParams Execution parameters
       */
      template<typename SrcDstT, typename ExecParamsT>
      void execute(SrcDstT* srcDst, const ExecParamsT& execParams)
      {
        static_assert(isKnownType<SrcDstT>, "A known type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecTypes(typePrecision<SrcDstT>, typeComplexity<SrcDstT>);
        executeUnsafe(srcDst, execParams);
      }

      /**
       * @brief Execute in-place plan
       * @tparam SrcDstT Source or destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param srcDst Source and destination buffer
       * @param execParams Execution parameters
       */
      template<typename SrcDstT, typename ExecParamsT>
      void execute(PlanarComplex<SrcDstT*> srcDst, const ExecParamsT& execParams)
      {
        static_assert(isRealType<SrcDstT>, "A real type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecTypes(typePrecision<SrcDstT>, Complexity::complex);
        executeUnsafe(srcDst, execParams);
      }

      /**
       * @brief Execute the plan
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void execute(SrcT* src, DstT* dst, const ExecParamsT& execParams)
      {
        static_assert(isKnownType<SrcT>, "A known type is required");
        static_assert(isKnownType<DstT>, "A known type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecTypes(typePrecision<SrcT>, typeComplexity<SrcT>, typePrecision<DstT>, typeComplexity<DstT>);
        executeUnsafe(src, dst, execParams);
      }

      /**
       * @brief Execute the plan with source buffer as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer in PlanarComplex format
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void execute(PlanarComplex<SrcT*> src, DstT* dst, const ExecParamsT& execParams)
      {
        static_assert(isRealType<SrcT>, "A real type is required");
        static_assert(isKnownType<DstT>, "A known type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecTypes(typePrecision<SrcT>, Complexity::complex, typePrecision<DstT>, typeComplexity<DstT>);
        executeUnsafe(src, dst, execParams);
      }

      /**
       * @brief Execute the plan with destination buffer as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer in PlanarComplex format
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void execute(SrcT* src, PlanarComplex<DstT*> dst, const ExecParamsT& execParams)
      {
        static_assert(isKnownType<SrcT>, "A known type is required");
        static_assert(isRealType<DstT>, "A real type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecTypes(typePrecision<SrcT>, typeComplexity<SrcT>, typePrecision<DstT>, Complexity::complex);
        executeUnsafe(src, dst, execParams);
      }

      /**
       * @brief Execute the plan with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer in PlanarComplex format
       * @param dst Destination buffer in PlanarComplex format
       * @param execParams Execution parameters
       */
      template<typename SrcT, typename DstT, typename ExecParamsT>
      void execute(PlanarComplex<SrcT*> src, PlanarComplex<DstT*> dst, const ExecParamsT& execParams)
      {
        static_assert(isRealType<SrcT>, "A real type is required");
        static_assert(isRealType<DstT>, "A real type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecTypes(typePrecision<SrcT>, Complexity::complex, typePrecision<DstT>, Complexity::complex);
        executeUnsafe(src, dst, execParams);
      }

      /**
       * @brief Execute the plan with default target parameters
       * @tparam ArgsT Argument types
       * @param args Arguments
       */
      template<typename... Args>
      void executeUnsafeWithDefaultTargetParameters(Args&&... args)
      {
        switch (getConfig().getTarget())
        {
        case Target::cpu:
          executeUnsafe(std::forward<Args>(args)..., afft::cpu::Parameters{});
          break;
        case Target::gpu:
          executeUnsafe(std::forward<Args>(args)..., afft::gpu::Parameters{});
          break;
        default:
          cxx::unreachable();
        }
      }

      /**
       * @brief Execute in-place plan
       * @tparam SrcDstT Source or destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param srcDst Source and destination buffer
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(void* srcDst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecParameters(execParams);
        requireInPlaceTransform();
        requireNotNull(srcDst);

// #     if AFFT_GPU_FRAMEWORK_IS(OPENCL)
//         if (std::is_same_v<ExecParamsT, afft::gpu::Parameters>)
//         {
//           const auto& gpuConfig = getConfig().getTargetConfig<Target::gpu>();
//           auto clSrcDst = gpu::opencl::makeBufferFromPtr(gpuConfig.context, srcDst, mConfig.getSrcShapeVolume());
//           executeImpl(clSrcDst.get(), srcDst.get(), execParams);
//         }
//         else
// #     endif
        {
          executeImpl(srcDst, srcDst, execParams);
        }
      }

      /**
       * @brief Execute in-place plan
       * @tparam SrcDstT Source or destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param srcDst Source and destination buffer
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(PlanarComplex<void*> srcDst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecParameters(execParams);
        requireInPlaceTransform();
        requireNotNull(srcDst.real, srcDst.imag);
        executeImpl(srcDst, srcDst, execParams);
      }

      /**
       * @brief Execute the plan without type checking
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(void* src, void* dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        if (src == dst)
        {
          executeUnsafe(src, execParams);
        }
        else
        {
          checkExecParameters(execParams);
          requireOutOfPlaceTransform();
          requireNotNull(src, dst);
          executeImpl(src, dst, execParams);
        }
      }

      /**
       * @brief Execute the plan without type checking
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(const void* src, void* dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecParameters(execParams);
        requireNonDestructiveTransform();
        requireOutOfPlaceTransform();
        requireNotNull(src, dst);
        executeImpl(removeConstFromPtr(src), dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking and with source buffer as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(PlanarComplex<void*> src, void* dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        if (src.real == dst)
        {
          executeUnsafe(src.real, execParams);
        }
        else
        {
          checkExecParameters(execParams);
          requireOutOfPlaceTransform();
          requireNotNull(src.real, src.imag, dst);
          executeImpl(src, dst, execParams);
        }
      }

      /**
       * @brief Execute the plan without type checking and with source buffer as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(PlanarComplex<const void*> src, void* dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecParameters(execParams);
        requireNonDestructiveTransform();
        requireOutOfPlaceTransform();
        requireNotNull(src.real, src.imag, dst);
        executeUnsafe(PlanarComplex<void*>{removeConstFromPtr(src.real),
                                           removeConstFromPtr(src.imag)}, dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(void* src, PlanarComplex<void*> dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        if (src == dst.real)
        {
          executeUnsafe(dst, execParams);
        }
        else
        {
          checkExecParameters(execParams);
          requireOutOfPlaceTransform();
          requireNotNull(src, dst.real, dst.imag);
          executeImpl(src, dst, execParams);
        }
      }

      /**
       * @brief Execute the plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(const void* src, PlanarComplex<void*> dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecParameters(execParams);
        requireNonDestructiveTransform();
        requireOutOfPlaceTransform();
        requireNotNull(src, dst.real, dst.imag);
        executeUnsafe(removeConstFromPtr(src), dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(PlanarComplex<void*> src, PlanarComplex<void*> dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        if (src.real == dst.real && src.imag == dst.imag)
        {
          executeUnsafe(src, execParams);
        }
        else
        {
          checkExecParameters(execParams);
          requireOutOfPlaceTransform();
          requireNotNull(src.real, src.imag, dst.real, dst.imag);
          executeImpl(src, dst, execParams);
        }
      }

      /**
       * @brief Execute the plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffer
       * @param dst Destination buffer
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(PlanarComplex<const void*> src, PlanarComplex<void*> dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecParameters(execParams);
        requireNonDestructiveTransform();
        requireOutOfPlaceTransform();
        requireNotNull(src.real, src.imag, dst.real, dst.imag);
        executeUnsafe(PlanarComplex<void*>{removeConstFromPtr(src.real),
                                           removeConstFromPtr(src.imag)}, dst, execParams);
      }

#   if AFFT_GPU_FRAMEWORK_IS(OPENCL)
      template<typename ExecParamsT>
      void executeUnsafe(cl_mem srcDst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecParameters(execParams);
        requireInPlaceTransform();
        requireNotNull(srcDst);

        if (gpu::opencl::isReadOnlyBuffer(srcDst))
        {
          throw makeException<std::invalid_argument>("Read-only buffer passed to in-place execute()");
        }

        executeImpl(srcDst, srcDst, execParams);
      }

      template<typename ExecParamsT>
      void executeUnsafe(PlanarComplex<cl_mem> srcDst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecParameters(execParams);
        requireInPlaceTransform();
        requireNotNull(srcDst.real, srcDst.imag);

        if (gpu::opencl::isReadOnlyBuffer(srcDst.real))
        {
          throw makeException<std::invalid_argument>("Read-only buffer passed to in-place execute()");
        }

        if (gpu::opencl::isReadOnlyBuffer(srcDst.imag))
        {
          throw makeException<std::invalid_argument>("Read-only buffer passed to in-place execute()");
        }

        executeImpl(srcDst, srcDst, execParams);
      }

      template<typename ExecParamsT>
      void executeUnsafe(cl_mem src, cl_mem dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        if (src == dst)
        {
          executeUnsafe(src, execParams);
        }
        else
        {
          checkExecParameters(execParams);
          requireOutOfPlaceTransform();
          requireNotNull(src, dst);

          if (gpu::opencl::isReadOnlyBuffer(src))
          {
            requireNonDestructiveTransform();
          }

          if (gpu::opencl::isReadOnlyBuffer(dst))
          {
            throw makeException<std::invalid_argument>("Read-only buffer passed as destination to execute()");
          }

          executeImpl(src, dst, execParams);
        }
      }

      template<typename ExecParamsT>
      void executeUnsafe(PlanarComplex<cl_mem> src, cl_mem dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        if (src.real == dst)
        {
          executeUnsafe(src, execParams);
        }
        else
        {
          checkExecParameters(execParams);
          requireOutOfPlaceTransform();
          requireNotNull(src.real, src.imag, dst);

          if (gpu::opencl::isReadOnlyBuffer(src.real) || gpu::opencl::isReadOnlyBuffer(src.imag))
          {
            requireNonDestructiveTransform();
          }

          if (gpu::opencl::isReadOnlyBuffer(dst))
          {
            throw makeException<std::invalid_argument>("Read-only buffer passed as destination to execute()");
          }

          executeImpl(src, dst, execParams);
        }
      }

      template<typename ExecutionParamsT>
      void executeUnsafe(cl_mem src, PlanarComplex<cl_mem> dst, const ExecutionParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        if (src == dst.real)
        {
          executeUnsafe(dst, execParams);
        }
        else
        {
          checkExecParameters(execParams);
          requireOutOfPlaceTransform();
          requireNotNull(src, dst.real, dst.imag);

          if (gpu::opencl::isReadOnlyBuffer(src))
          {
            requireNonDestructiveTransform();
          }

          if (gpu::opencl::isReadOnlyBuffer(dst.real) || gpu::opencl::isReadOnlyBuffer(dst.imag))
          {
            throw makeException<std::invalid_argument>("Read-only buffer passed as destination to execute()");
          }

          executeImpl(src, dst, execParams);
        }
      }

      template<typename ExecutionParams>
      void executeUnsafe(PlanarComplex<cl_mem> src, PlanarComplex<cl_mem> dst, const ExecutionParams& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        if (src.real == dst.real && src.imag == dst.imag)
        {
          executeUnsafe(src, execParams);
        }
        else
        {
          checkExecParameters(execParams);
          requireOutOfPlaceTransform();
          requireNotNull(src.real, src.imag, dst.real, dst.imag);

          if (gpu::opencl::isReadOnlyBuffer(src.real) || gpu::opencl::isReadOnlyBuffer(src.imag))
          {
            requireNonDestructiveTransform();
          }

          if (gpu::opencl::isReadOnlyBuffer(dst.real) || gpu::opencl::isReadOnlyBuffer(dst.imag))
          {
            throw makeException<std::invalid_argument>("Read-only buffer passed as destination to execute()");
          }

          executeImpl(src, dst, execParams);
        }
      }
#   endif

      /**
       * @brief Get the size of the workspace required for the plan
       * @return std::size_t the size of the workspace in bytes
       */
      virtual std::size_t getWorkspaceSize() const { return {}; }

      /**
       * @brief Check if the non-destructive transform is configured
       */
      constexpr void requireNonDestructiveTransform() const
      {
        if (mConfig.getCommonParameters().destroySource)
        {
          throw makeException<std::runtime_error>("Running a destructive transform on const input data.");
        }
      }

      /**
       * @brief Check if the out-of-place transform is configured
       */
      constexpr void requireOutOfPlaceTransform() const
      {
        if (mConfig.getCommonParameters().placement != Placement::outOfPlace)
        {
          throw makeException<std::runtime_error>("Running an in-place transform with out-of-place data.");
        }
      }

      /**
       * @brief Check if the in-place transform is configured
       */
      constexpr void requireInPlaceTransform() const
      {
        if (mConfig.getCommonParameters().placement != Placement::inPlace)
        {
          throw makeException<std::runtime_error>("Running an out-of-place transform with in-place data.");
        }
      }

      /**
       * @brief Check if the execution parameters are valid
       * @param execParams the execution parameters
       */
      constexpr void checkExecParameters(const afft::cpu::ExecutionParameters&) const
      {
        if (mConfig.getTarget() != Target::cpu)
        {
          throw makeException<std::invalid_argument>("CPU execution parameters passed to a non-CPU plan");
        }
      }

      /**
       * @brief Check if the execution parameters are valid
       * @param execParams the execution parameters
       */
      constexpr void checkExecParameters(const afft::gpu::ExecutionParameters&) const
      {
        if (mConfig.getTarget() != Target::gpu)
        {
          throw makeException<std::invalid_argument>("GPU execution parameters passed to a non-GPU plan");
        }
      }

      /**
       * @brief Check execution type for in-place transform
       * @param srcOrDstPrec the source or destination precision
       * @param srcOrDstCmpl the source or destination complexity
       */
      constexpr void checkExecTypes(Precision srcOrDstPrec, Complexity srcOrDstCmpl) const
      {
        const auto& prec = mConfig.getTransformPrecision();

        if (srcOrDstPrec != prec.source && srcOrDstPrec != prec.destination)
        {
          throw makeException<std::invalid_argument>("Invalid precision for transform");
        }

        const auto [refSrcCmpl, refDstCmpl] = mConfig.getSrcDstComplexity();

        if (srcOrDstCmpl != refSrcCmpl && srcOrDstCmpl != refDstCmpl)
        {
          throw makeException<std::invalid_argument>("Invalid complexity for transform");
        }
      }

      /**
       * @brief Check if the execution types are valid
       * @param srcPrec the source precision
       * @param srcCmpl the source complexity
       * @param dstPrec the destination precision
       * @param dstCmpl the destination complexity
       */
      constexpr void checkExecTypes(Precision srcPrec, Complexity srcCmpl, Precision dstPrec, Complexity dstCmpl) const
      {
        const auto& prec = mConfig.getTransformPrecision();

        if (srcPrec != prec.source)
        {
          throw makeException<std::invalid_argument>("Invalid source precision for transform");
        }
        
        if (dstPrec != prec.destination)
        {
          throw makeException<std::invalid_argument>("Invalid destination precision for transform");
        }

        const auto [refSrcCmpl, refDstCmpl] = mConfig.getSrcDstComplexity();

        if (srcCmpl != refSrcCmpl)
        {
          throw makeException<std::invalid_argument>("Invalid source complexity for transform");
        }

        if (dstCmpl != refDstCmpl)
        {
          throw makeException<std::invalid_argument>("Invalid destination complexity for transform");
        }
      }
    protected:
      /**
       * @brief Construct a new PlanImpl object
       * @param config the configuration of the plan
       */
      PlanImpl(const Config& config) noexcept
      : mConfig(config)
      {}

      /**
       * @brief Implementation of the plan execution on the CPU
       * @param src the source buffer
       * @param dst the destination buffer
       * @param execParams the CPU execution parameters
       */
      virtual void executeImpl(ExecParam, ExecParam, const afft::cpu::ExecutionParameters&)
      {
        throw makeException<std::runtime_error>("CPU execution is by currently selected implementation");
      }

      /**
       * @brief Implementation of the plan execution on the GPU
       * @param src the source buffer
       * @param dst the destination buffer
       * @param execParams the GPU execution parameters
       */
      virtual void executeImpl(ExecParam, ExecParam, const afft::gpu::ExecutionParameters&)
      {
        throw makeException<std::runtime_error>("GPU execution is by currently selected implementation");
      }

    private:
      /**
       * @brief Require that the pointers are not null
       * @param ptrs the pointers to check
       */
      template<typename... T>
      void requireNotNull(const T*... ptrs)
      {
        if ((false || ... || (ptrs == nullptr)))
        {
          throw makeException<std::invalid_argument>("Null pointer passed to execute()");
        }
      }

      Config mConfig; ///< The configuration of the plan
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_PLAN_IMPL_HPP */
