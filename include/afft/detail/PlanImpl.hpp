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

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "cxx.hpp"
#include "Desc.hpp"

namespace afft::detail
{
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
       * @brief Get the description of the plan
       * @return const reference to the description of the plan
       */
      [[nodiscard]] constexpr const Desc& getDesc() const noexcept
      {
        return mDesc;
      }

      /**
       * @brief Get the backend of the plan
       * @return Backend of the plan
       */
      [[nodiscard]] virtual Backend getBackend() const = 0;

      /**
       * @brief Execute in-place plan
       * @tparam SrcDstT Source or destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param srcDst Source and destination buffer
       * @param execParams Execution parameters
       */
      template<typename SrcDstT, typename ExecParamsT>
      void execute(View<SrcDstT*> srcDst, const ExecParamsT& execParams)
      {
        static_assert(isKnownType<SrcDstT>, "A known type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecTypes(typePrecision<SrcDstT>, typeComplexity<SrcDstT>);
        executeUnsafe(reinterpretViewCast<void*>(srcDst), execParams);
      }

      /**
       * @brief Execute in-place plan
       * @tparam SrcDstT Source or destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param srcDst Source and destination buffer
       * @param execParams Execution parameters
       */
      template<typename SrcDstT, typename ExecParamsT>
      void execute(View<PlanarComplex<SrcDstT*>> srcDst, const ExecParamsT& execParams)
      {
        static_assert(isRealType<SrcDstT>, "A real type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecTypes(typePrecision<SrcDstT>, Complexity::complex);
        executeUnsafe(reinterpretViewCast<PlanarComplex<void*>>(srcDst), execParams);
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
      void execute(View<SrcT*> src, View<DstT*> dst, const ExecParamsT& execParams)
      {
        static_assert(isKnownType<SrcT>, "A known type is required");
        static_assert(isKnownType<DstT>, "A known type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        using SrcVoidT = std::conditional_t<std::is_const_v<SrcT>, const void, void>;

        checkExecTypes(typePrecision<SrcT>, typeComplexity<SrcT>, typePrecision<DstT>, typeComplexity<DstT>);
        executeUnsafe(reinterpretViewCast<SrcVoidT*>(src), reinterpretViewCast<void*>(dst), execParams);
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
      void execute(View<PlanarComplex<SrcT*>> src, View<DstT*> dst, const ExecParamsT& execParams)
      {
        static_assert(isRealType<SrcT>, "A real type is required");
        static_assert(isKnownType<DstT>, "A known type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        using SrcVoidT = std::conditional_t<std::is_const_v<SrcT>, const void, void>;

        checkExecTypes(typePrecision<SrcT>, Complexity::complex, typePrecision<DstT>, typeComplexity<DstT>);
        executeUnsafe(reinterpretViewCast<PlanarComplex<SrcVoidT*>>(src), reinterpretViewCast<void*>(dst), execParams);
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
      void execute(View<SrcT*> src, View<PlanarComplex<DstT*>> dst, const ExecParamsT& execParams)
      {
        static_assert(isKnownType<SrcT>, "A known type is required");
        static_assert(isRealType<DstT>, "A real type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        using SrcVoidT = std::conditional_t<std::is_const_v<SrcT>, const void, void>;

        checkExecTypes(typePrecision<SrcT>, typeComplexity<SrcT>, typePrecision<DstT>, Complexity::complex);
        executeUnsafe(reinterpretViewCast<SrcVoidT*>(src), reinterpretViewCast<PlanarComplex<void*>>(dst), execParams);
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
      void execute(View<PlanarComplex<SrcT*>> src, View<PlanarComplex<DstT*>> dst, const ExecParamsT& execParams)
      {
        static_assert(isRealType<SrcT>, "A real type is required");
        static_assert(isRealType<DstT>, "A real type is required");
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        using SrcVoidT = std::conditional_t<std::is_const_v<SrcT>, const void, void>;

        checkExecTypes(typePrecision<SrcT>, Complexity::complex, typePrecision<DstT>, Complexity::complex);
        executeUnsafe(reinterpretViewCast<PlanarComplex<SrcVoidT*>>(src),
                      reinterpretViewCast<PlanarComplex<void*>>(dst),
                      execParams);
      }

      /**
       * @brief Execute in-place plan
       * @tparam SrcDstT Source or destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param srcDst Source and destination buffer
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(View<void*> srcDst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkBuffers(srcDst);
        checkExecParameters(execParams);
        requireInPlaceTransform();

// #     if AFFT_GPU_BACKEND_IS(OPENCL)
//         if (std::is_same_v<ExecParamsT, afft::gpu::Parameters>)
//         {
//           const auto& gpuConfig = getDesc().getTargetConfig<Target::gpu>();
//           auto clSrcDst = gpu::opencl::makeBufferFromPtr(gpuConfig.context, srcDst, mDesc.getSrcShapeVolume());
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
      void executeUnsafe(View<PlanarComplex<void*>> srcDst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkBuffers(srcDst);
        checkExecParameters(execParams);
        requireInPlaceTransform();
        executeImpl(srcDst, srcDst, execParams);
      }

      /**
       * @brief Execute the plan without type checking
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers
       * @param dst Destination buffers
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(View<void*> src, View<void*> dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkBuffers(src);
        checkBuffers(dst);

        if (std::equal(src.begin(), src.end(), dst.begin()))
        {
          executeUnsafe(src, execParams);
        }
        else
        {
          checkExecParameters(execParams);
          requireOutOfPlaceTransform();
          executeImpl(src, dst, execParams);
        }
      }

      /**
       * @brief Execute the plan without type checking
       * @tparam SrcT Source buffer type
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers
       * @param dst Destination buffers
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(View<const void*> src, View<void*> dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkBuffers(src);
        checkBuffers(dst);

        checkExecParameters(execParams);
        requireNonDestructiveTransform();
        requireOutOfPlaceTransform();
        executeImpl(reinterpretViewCast<void*>(src), dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking and with source buffer as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers
       * @param dst Destination buffers
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(View<PlanarComplex<void*>> src, View<void*> dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkBuffers(src);
        checkBuffers(dst);

        if (std::equal(src.begin(), src.end(), dst.begin(), [](const auto& s, const auto& d){ return s.real == d; }))
        {
          executeUnsafe(src, execParams);
        }
        else
        {
          checkExecParameters(execParams);
          requireOutOfPlaceTransform();
          executeImpl(src, dst, execParams);
        }
      }

      /**
       * @brief Execute the plan without type checking and with source buffer as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers
       * @param dst Destination buffers
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(View<PlanarComplex<const void*>> src, View<void*> dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkBuffers(src);
        checkBuffers(dst);

        checkExecParameters(execParams);
        requireNonDestructiveTransform();
        requireOutOfPlaceTransform();
        executeUnsafe(reinterpretViewCast<PlanarComplex<void*>>(src), dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers
       * @param dst Destination buffers
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(View<void*> src, View<PlanarComplex<void*>> dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkBuffers(src);
        checkBuffers(dst);

        if (std::equal(src.begin(), src.end(), dst.begin(), [](const auto& s, const auto& d){ return s == d.real; }))
        {
          executeUnsafe(dst, execParams);
        }
        else
        {
          checkExecParameters(execParams);
          requireOutOfPlaceTransform();
          executeImpl(src, dst, execParams);
        }
      }

      /**
       * @brief Execute the plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers
       * @param dst Destination buffers
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(View<const void*> src, View<PlanarComplex<void*>> dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkBuffers(src);
        checkBuffers(dst);

        checkExecParameters(execParams);
        requireNonDestructiveTransform();
        requireOutOfPlaceTransform();
        executeUnsafe(reinterpretViewCast<void*>(src), dst, execParams);
      }

      /**
       * @brief Execute the plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers
       * @param dst Destination buffers
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(View<PlanarComplex<void*>> src, View<PlanarComplex<void*>> dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkBuffers(src);
        checkBuffers(dst);

        if (std::equal(src.begin(), src.end(), dst.begin(), [](const auto& s, const auto& d)
                                                            { return s.real == d.real && s.imag == d.imag; }))
        {
          executeUnsafe(src, execParams);
        }
        else
        {
          checkExecParameters(execParams);
          requireOutOfPlaceTransform();
          executeImpl(src, dst, execParams);
        }
      }

      /**
       * @brief Execute the plan without type checking and with source and destination buffers as PlanarComplex
       * @tparam SrcT Source buffer type in PlanarComplex format
       * @tparam DstT Destination buffer type in PlanarComplex format
       * @tparam ExecParamsT Execution parameters type
       * @param src Source buffers
       * @param dst Destination buffers
       * @param execParams Execution parameters
       */
      template<typename ExecParamsT>
      void executeUnsafe(View<PlanarComplex<const void*>> src, View<PlanarComplex<void*>> dst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkBuffers(src);
        checkBuffers(dst);

        checkExecParameters(execParams);
        requireNonDestructiveTransform();
        requireOutOfPlaceTransform();
        executeUnsafe(reinterpretViewCast<PlanarComplex<void*>>(src), dst, execParams);
      }

#   if AFFT_GPU_BACKEND_IS(OPENCL)
      template<typename ExecParamsT>
      void executeUnsafe(cl_mem srcDst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecParameters(execParams);
        requireInPlaceTransform();

        if (gpu::opencl::isReadOnlyBuffer(srcDst))
        {
          throw std::invalid_argument{"Read-only buffer passed to in-place execute()"};
        }

        executeImpl(srcDst, srcDst, execParams);
      }

      template<typename ExecParamsT>
      void executeUnsafe(PlanarComplex<cl_mem> srcDst, const ExecParamsT& execParams)
      {
        static_assert(isExecutionParameters<ExecParamsT>, "Unknown execution parameters type");

        checkExecParameters(execParams);
        requireInPlaceTransform();

        if (gpu::opencl::isReadOnlyBuffer(srcDst.real))
        {
          throw std::invalid_argument{"Read-only buffer passed to in-place execute()"};
        }

        if (gpu::opencl::isReadOnlyBuffer(srcDst.imag))
        {
          throw std::invalid_argument{"Read-only buffer passed to in-place execute()"};
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

          if (gpu::opencl::isReadOnlyBuffer(src))
          {
            requireNonDestructiveTransform();
          }

          if (gpu::opencl::isReadOnlyBuffer(dst))
          {
            throw std::invalid_argument{"Read-only buffer passed as destination to execute()"};
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

          if (gpu::opencl::isReadOnlyBuffer(src.real) || gpu::opencl::isReadOnlyBuffer(src.imag))
          {
            requireNonDestructiveTransform();
          }

          if (gpu::opencl::isReadOnlyBuffer(dst))
          {
            throw std::invalid_argument{"Read-only buffer passed as destination to execute()"};
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

          if (gpu::opencl::isReadOnlyBuffer(src))
          {
            requireNonDestructiveTransform();
          }

          if (gpu::opencl::isReadOnlyBuffer(dst.real) || gpu::opencl::isReadOnlyBuffer(dst.imag))
          {
            throw std::invalid_argument{"Read-only buffer passed as destination to execute()"};
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

          if (gpu::opencl::isReadOnlyBuffer(src.real) || gpu::opencl::isReadOnlyBuffer(src.imag))
          {
            requireNonDestructiveTransform();
          }

          if (gpu::opencl::isReadOnlyBuffer(dst.real) || gpu::opencl::isReadOnlyBuffer(dst.imag))
          {
            throw std::invalid_argument{"Read-only buffer passed as destination to execute()"};
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
       * @brief Check if the buffer count matches the target count and if any buffer is null
       * @param count the buffer count
       */
      template<typename T>
      void checkBuffers(View<T*> buffers) const
      {
        auto bufferIsNotNull = [](const auto& buffer){ return buffer != nullptr; };

        if (buffers.size() != mDesc.getTargetCount())
        {
          throw std::invalid_argument{"Invalid number of buffers passed to execute()"};
        }

        if (std::any_of(buffers.begin(), buffers.end(), bufferIsNotNull))
        {
          throw std::invalid_argument{"Null pointer passed to execute()"};
        }
      }

      /**
       * @brief Check if the buffer count matches the target count and if any buffer is null
       * @param count the buffer count
       */
      template<typename T>
      void checkBuffers(View<PlanarComplex<T*>> buffers) const
      {
        auto bufferIsNotNull = [](const auto& buffer){ return buffer.real != nullptr && buffer.imag != nullptr; };

        if (buffers.size() != mDesc.getTargetCount())
        {
          throw std::invalid_argument{"Invalid number of buffers passed to execute()"};
        }

        if (std::any_of(buffers.begin(), buffers.end(), bufferIsNotNull))
        {
          throw std::invalid_argument{"Null pointer passed to execute()"};
        }
      }

      /// @brief Require interleaved complex format
      void requireInterleavedComplex() const
      {
        if (mDesc.getComplexFormat() != ComplexFormat::interleaved)
        {
          throw std::runtime_error{"Plan expects interleaved complex format"};
        }
      }

      /// @brief Require planar complex format
      void requirePlanarComplex() const
      {
        if (mDesc.getComplexFormat() != ComplexFormat::planar)
        {
          throw std::runtime_error{"Plan expects planar complex format"};
        }
      }

      /**
       * @brief Check if the non-destructive transform is configured
       */
      constexpr void requireNonDestructiveTransform() const
      {
        if (!mDesc.getPreserveSource())
        {
          throw std::runtime_error{"Running a destructive transform on const input data."};
        }
      }

      /**
       * @brief Check if the out-of-place transform is configured
       */
      constexpr void requireOutOfPlaceTransform() const
      {
        if (mDesc.getPlacement() != Placement::outOfPlace)
        {
          throw std::runtime_error{"Running an in-place transform with out-of-place data."};
        }
      }

      /**
       * @brief Check if the in-place transform is configured
       */
      constexpr void requireInPlaceTransform() const
      {
        if (mDesc.getPlacement() != Placement::inPlace)
        {
          throw std::runtime_error{"Running an out-of-place transform with in-place data."};
        }
      }

      /**
       * @brief Check if the execution parameters are valid
       * @tparam ExecutionParametersT the execution parameters type
       * @param execParams the execution parameters
       */
      template<typename ExecutionParametersT>
      constexpr void checkExecParameters(const ExecutionParametersT& execParams) const
      {
        if (execParams.target != mDesc.getTarget())
        {
          throw std::invalid_argument{"Invalid target for transform"};
        }

        if (execParams.distribution != mDesc.getDistribution())
        {
          throw std::invalid_argument{"Invalid distribution for transform"};
        }
      }

      /**
       * @brief Check execution type for in-place transform
       * @param srcOrDstPrec the source or destination precision
       * @param srcOrDstCmpl the source or destination complexity
       */
      constexpr void checkExecTypes(Precision srcOrDstPrec, Complexity srcOrDstCmpl) const
      {
        const auto& prec = mDesc.getPrecision();

        if (srcOrDstPrec != prec.source && srcOrDstPrec != prec.destination)
        {
          throw std::invalid_argument{"Invalid precision for transform"};
        }

        const auto [refSrcCmpl, refDstCmpl] = mDesc.getSrcDstComplexity();

        if (srcOrDstCmpl != refSrcCmpl && srcOrDstCmpl != refDstCmpl)
        {
          throw std::invalid_argument{"Invalid complexity for transform"};
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
        const auto& prec = mDesc.getPrecision();

        if (srcPrec != prec.source)
        {
          throw std::invalid_argument{"Invalid source precision for transform"};
        }
        
        if (dstPrec != prec.destination)
        {
          throw std::invalid_argument{"Invalid destination precision for transform"};
        }

        const auto [refSrcCmpl, refDstCmpl] = mDesc.getSrcDstComplexity();

        if (srcCmpl != refSrcCmpl)
        {
          throw std::invalid_argument{"Invalid source complexity for transform"};
        }

        if (dstCmpl != refDstCmpl)
        {
          throw std::invalid_argument{"Invalid destination complexity for transform"};
        }
      }
    protected:
      /**
       * @brief Construct a new PlanImpl object
       * @param config the configuration of the plan
       */
      PlanImpl(const Desc& desc)
      : mDesc(desc)
      {}

      /**
       * @brief Execute the plan
       * @param src the source buffer
       * @param dst the destination buffer
       * @param execParams the execution parameters
       */
      virtual void executeImpl(View<void*>, View<void*>, const afft::spst::cpu::ExecutionParameters&)
      {
        throw std::runtime_error{"backend does not implement spst cpu execution"};
      }

      /**
       * @brief Execute the plan
       * @param src the source buffer
       * @param dst the destination buffer
       * @param execParams the execution parameters
       */
      virtual void executeImpl(View<void*>, View<void*>, const afft::spst::gpu::ExecutionParameters&)
      {
        throw std::runtime_error{"backend does not implement spst gpu execution"};
      }

      /**
       * @brief Execute the plan
       * @param src the source buffer
       * @param dst the destination buffer
       * @param execParams the execution parameters
       */
      virtual void executeImpl(View<void*>, View<void*>, const afft::spmt::gpu::ExecutionParameters&)
      {
        throw std::runtime_error{"backend does not implement spmt gpu execution"};
      }

      /**
       * @brief Execute the plan
       * @param src the source buffer
       * @param dst the destination buffer
       * @param execParams the execution parameters
       */
      virtual void executeImpl(View<void*>, View<void*>, const afft::mpst::cpu::ExecutionParameters&)
      {
        throw std::runtime_error{"backend does not implement mpst cpu execution"};
      }

      /**
       * @brief Execute the plan
       * @param src the source buffer
       * @param dst the destination buffer
       * @param execParams the execution parameters
       */
      virtual void executeImpl(View<void*>, View<void*>, const afft::mpst::gpu::ExecutionParameters&)
      {
        throw std::runtime_error{"backend does not implement mpst gpu execution"};
      }

    private:
      Desc mDesc; ///< The description of the plan
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_PLAN_IMPL_HPP */
