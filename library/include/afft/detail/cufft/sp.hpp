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

#ifndef AFFT_DETAIL_CUFFT_SP_HPP
#define AFFT_DETAIL_CUFFT_SP_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../Plan.hpp"

namespace afft::detail::cufft::sp
{
  /**
   * @brief Create a cufft sp plan.
   * @param desc The plan descriptor.
   * @param backendParams The backend parameters.
   * @return The plan.
   */
  [[nodiscard]] std::unique_ptr<afft::Plan>
  makePlan(const Description& desc, const afft::cuda::BackendParameters& backendParams);
} // namespace afft::detail::cufft::sp

#ifdef AFFT_HEADER_ONLY

#include "common.hpp"
#include "error.hpp"
#include "Plan.hpp"

namespace afft::detail::cufft::sp
{
  /// @brief The maximum number of targets for sp cufft transforms.
  inline constexpr std::size_t maxTargetCount = 16;

  /**
   * @class PlanImpl
   * @brief Implementation of the sp plan interface for cuFFT.
   */
  class SingleDevicePlan final : public cufft::Plan<MpBackend::none>
  {
    private:
      /// @brief Alias for the parent class.
      using Parent = cufft::Plan<MpBackend::none>;

    public:
      /// @brief inherit constructors
      using Parent::Parent;

      /**
       * @brief Constructor.
       * @param desc The plan descriptor.
       * @param cufftParams The cuFFT parameters.
       */
      SingleDevicePlan(const Description& desc, const afft::cuda::BackendParameters& backendParams)
      : Parent{desc, backendParams}
      {
        const int device = Parent::mDesc.template getTargetDesc<Target::cuda>().getDevices()[0];

        cuda::ScopedDevice scopedDevice{device};

        if (Parent::mBackendParams.cufft.usePatientJit)
        {
#       if CUFFT_VERSION >= 11200
          checkError(cufftSetPlanPropertyInt64(mHandle, NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT, 1));
#       endif
        }

        // if (desc.useExternalWorkspace())
        // {
        //   checkError(cufftSetAutoAllocation(mHandle, 0));
        // }

//         if (dftDesc.type == dft::Type::complexToComplex && std::all_of(n.begin(), n.end(), [](auto size){ return size <= 4096}))
//         {
// #       if CUFFT_VERSION >= 9200
//           checkError(cufftXtSetWorkAreaPolicy(planImpl->mHandle,
//                                               makeWorkAreaPolicy(cufftParams.workspacePolicy)),
//                                               &cufftParams.userWorkspaceSize);
// #       endif
//        }

        const auto precision          = Parent::mDesc.getPrecision().execution;
        const auto [srcCmpl, dstCmpl] = Parent::mDesc.getSrcDstComplexity();

        const auto& memDesc = Parent::mDesc.template getMemDesc<MemoryLayout::centralized>();

        const auto shapeRank     = Parent::mDesc.getShapeRank();
        const auto transformRank = Parent::mDesc.getTransformRank();
        const auto transformAxes = Parent::mDesc.getTransformAxes();
        const auto srcShape      = Parent::mDesc.getSrcShape();
        const auto dstShape      = Parent::mDesc.getDstShape();
        const auto srcStrides    = memDesc.getSrcStrides();
        const auto dstStrides    = memDesc.getDstStrides();

        auto n                  = Parent::mDesc.template getTransformDimsAs<SizeT>();
        auto srcNEmbedAndStride = makeNEmbedAndStride<SizeT>({srcShape.data, shapeRank},
                                                             transformAxes,
                                                             srcStrides);
        auto dstNEmbedAndStride = makeNEmbedAndStride<SizeT>({dstShape.data, shapeRank},
                                                             transformAxes,
                                                             dstStrides);

        SizeT batch{1};
        SizeT srcDist{1};
        SizeT dstDist{1};

        if (const auto howManyRank = Parent::mDesc.getTransformHowManyRank(); howManyRank == 1)
        {
          const auto howManyAxis = mDesc.getTransformHowManyAxes().front();

          batch   = safeIntCast<SizeT>(Parent::mDesc.getShape()[howManyAxis]);
          srcDist = safeIntCast<SizeT>(srcStrides[howManyAxis]);
          dstDist = safeIntCast<SizeT>(dstStrides[howManyAxis]);
        }
        else if (howManyRank > 1)
        {
          const auto shape       = Parent::mDesc.getShape();
          const auto howManyAxes = mDesc.getTransformHowManyAxes();

          batch   = shape[howManyAxes.front()];
          srcDist = safeIntCast<SizeT>(srcStrides[howManyAxes.back()]);
          dstDist = safeIntCast<SizeT>(dstStrides[howManyAxes.back()]);

          for (std::size_t i = howManyAxes.size() - 1; i > 0; --i)
          {
            if (howManyAxes[i] != howManyAxes[i - 1] + 1)
            {
              throw Exception{Error::cufft, "unsupported how many axes"};
            }

            if (srcStrides[howManyAxes[i]] * srcShape[howManyAxes[i]] != srcStrides[howManyAxes[i - 1]])
            {
              throw Exception{Error::cufft, "unsupported how many strides"};
            }

            if (dstStrides[howManyAxes[i]] * dstShape[howManyAxes[i]] != dstStrides[howManyAxes[i - 1]])
            {
              throw Exception{Error::cufft, "unsupported how many strides"};
            }

            batch *= safeIntCast<SizeT>(shape[howManyAxes[i]]);
          }
        }

#     if CUFFT_VERSION >= 11300
        if (const auto normalization = Parent::mDesc.getNormalization();
            normalization == Normalization::unitary || normalization == Normalization::orthogonal)
        {
          const auto complexity = dstCmpl;
          const auto normFactor = Parent::mDesc.template getNormalizationFactor<double>();

          const auto ltoirCode = makeNormalizationStoreCallbackCode(device, precision, complexity, normFactor);

          checkError(cufftXtSetJITCallback(mHandle,
                                           (complexity == Complexity::real) ? "storeReal" : "storeComplex",
                                           ltoirCode.data(),
                                           ltoirCode.size(),
                                           makeStoreCallbackType(precision, complexity),
                                           nullptr));
        }
#     endif

        checkError(cufftXtMakePlanMany(mHandle,
                                       static_cast<int>(transformRank),
                                       n.data,
                                       srcNEmbedAndStride.nEmbed.data,
                                       srcNEmbedAndStride.stride,
                                       srcDist,
                                       makeCudaDataType(precision, srcCmpl),
                                       dstNEmbedAndStride.nEmbed.data,
                                       dstNEmbedAndStride.stride,
                                       dstDist,
                                       makeCudaDataType(precision, dstCmpl),
                                       batch,
                                       &mWorkspaceSize,
                                       makeCudaDataType(precision, Complexity::complex)));
      }

      /// @brief Destructor.
      ~SingleDevicePlan() override = default;

      /// @brief Inherit assignment operator
      using Parent::operator=;

      /**
       * @brief Get element count of the source buffers.
       * @return Element count of the source buffers.
       */
      [[nodiscard]] View<std::size_t> getSrcElemCounts() const noexcept override
      {
        return makeScalarView(mSrcElemCount);
      }

      /**
       * @brief Get element count of the destination buffers.
       * @return Element count of the destination buffers.
       */
      [[nodiscard]] View<std::size_t> getDstElemCounts() const noexcept override
      {
        return makeScalarView(mDstElemCount);
      }

      /**
       * @brief Get the external workspace sizes
       * @return The workspace sizes
       */
      [[nodiscard]] View<std::size_t> getExternalWorkspaceSizes() const noexcept override
      {
        return makeScalarView(mWorkspaceSize);
      }

    private:
      /**
       * @brief Implementation of the executeImpl method.
       * @param src View of the source data pointers.
       * @param dst View of the destination data pointers.
       * @param execParams The execution parameters.
       */
      void executeBackendImpl(View<void*> src, View<void*> dst, const afft::cuda::ExecutionParameters& execParams) override
      {
        checkError(cufftSetStream(mHandle, execParams.stream));

        // if (getDesc().useExternalWorkspace())
        // {
        //   checkError(cufftSetWorkArea(mHandle, execParams.externalWorkspaces.front())); 
        // }

        checkError(cufftXtExec(mHandle, src.front(), dst.front(), getDirection()));
      }

      Handle      mHandle{};        ///< The cuFFT plan handle.
      std::size_t mWorkspaceSize{}; ///< The size of the workspace
      std::size_t mSrcElemCount{};  ///< The number of elements in the source buffer
      std::size_t mDstElemCount{};  ///< The number of elements in the destination buffer
  };

  /// @brief Plan for multi-GPU cuFFT transforms.
  class MultiDevicePlan final : public cufft::Plan<MpBackend::none>
  {
    private:
      /// @brief Alias for the parent class.
      using Parent = cufft::Plan<MpBackend::none>;

    public:
      /// @brief inherit constructors
      using Parent::Parent;

      /**
       * @brief Constructor.
       * @param desc The plan descriptor.
       * @param cufftParams The cuFFT parameters.
       */
      MultiDevicePlan(const Description& desc, const afft::cuda::BackendParameters& backendParams)
      : Parent{desc, backendParams}
      {
        throw Exception(Error::cufft, "multi-GPU transforms are not yet supported");

        const auto cudaDevices = Parent::mDesc.template getTargetDesc<Target::cuda>().getDevices();
        const auto targetCount = cudaDevices.size();

        checkError(cufftXtSetGPUs(mHandle, static_cast<int>(targetCount), const_cast<int*>(cudaDevices.data())));

        // // if (desc.useExternalWorkspace())
        // // {
        // //   checkError(cufftSetAutoAllocation(mHandle, 0));
        // // }

        const auto precision          = Parent::mDesc.getPrecision().execution;
        const auto [srcCmpl, dstCmpl] = Parent::mDesc.getSrcDstComplexity();

        const auto shapeRank     = Parent::mDesc.getShapeRank();
        const auto transformRank = Parent::mDesc.getTransformRank();
        const auto howManyRank   = Parent::mDesc.getTransformHowManyRank();
        const auto transformAxes = Parent::mDesc.getTransformAxes();
        const auto srcShape      = Parent::mDesc.getSrcShape<SizeT>();
        const auto dstShape      = Parent::mDesc.getDstShape<SizeT>();

        auto n = Parent::mDesc.template getTransformDimsAs<SizeT>();

        SizeT batch{1};
        SizeT srcDist{1};
        SizeT dstDist{1};

        if (const auto howManyRank = Parent::mDesc.getTransformHowManyRank(); howManyRank == 1)
        {
          const auto howManyAxis = mDesc.getTransformHowManyAxes().front();

          batch   = safeIntCast<SizeT>(Parent::mDesc.getShape()[howManyAxis]);
          srcDist = std::accumulate(srcShape.data, srcShape.data + transformRank, SizeT{1}, std::multiplies<>{});
          dstDist = std::accumulate(dstShape.data, dstShape.data + transformRank, SizeT{1}, std::multiplies<>{});
        }

        checkError(cufftXtMakePlanMany(mHandle,
                                       static_cast<int>(transformRank),
                                       n.data,
                                       nullptr,
                                       1,
                                       srcDist,
                                       makeCudaDataType(precision, srcCmpl),
                                       nullptr,
                                       1,
                                       dstDist,
                                       makeCudaDataType(precision, dstCmpl),
                                       batch,
                                       mWorkspaceSizes.data(),
                                       makeCudaDataType(precision, Complexity::complex)));

        // // TODO: set the src and dst target counts
    
        const auto srcElemSizeOf = Parent::mDesc.sizeOfSrcElem();
        const auto dstElemSizeOf = Parent::mDesc.sizeOfDstElem();

        mSrcDesc.version = CUDA_XT_DESCRIPTOR_VERSION;
        mSrcDesc.nGPUs   = static_cast<int>(targetCount);
        std::copy_n(cudaDevices.begin(), targetCount, mSrcDesc.GPUs);
        std::transform(mSrcElemCounts.begin(),
                       mSrcElemCounts.begin() + targetCount,
                       mSrcDesc.size,
                       [&](auto elemCount){ return elemCount * srcElemSizeOf; });

        mDstDesc.version = CUDA_XT_DESCRIPTOR_VERSION;
        mDstDesc.nGPUs   = static_cast<int>(targetCount);
        std::copy_n(cudaDevices.begin(), targetCount, mDstDesc.GPUs);
        std::transform(mDstElemCounts.begin(),
                       mDstElemCounts.begin() + targetCount,
                       mDstDesc.size,
                       [&](auto elemCount){ return elemCount * dstElemSizeOf; });

        // TODO: set the src and dst sub-formats
      }

      /// @brief Destructor.
      ~MultiDevicePlan() override = default;

      /// @brief Inherit assignment operator
      using Parent::operator=;

      /**
       * @brief Get element count of the source buffers.
       * @return Element count of the source buffers.
       */
      [[nodiscard]] View<std::size_t> getSrcElemCounts() const noexcept override
      {
        return {mSrcElemCounts.data(), Parent::mDesc.getTargetCount()};
      }

      /**
       * @brief Get element count of the destination buffers.
       * @return Element count of the destination buffers.
       */
      [[nodiscard]] View<std::size_t> getDstElemCounts() const noexcept override
      {
        return {mDstElemCounts.data(), Parent::mDesc.getTargetCount()};
      }

      /**
       * @brief Get the external workspace sizes
       * @return The workspace sizes
       */
      [[nodiscard]] View<std::size_t> getExternalWorkspaceSizes() const noexcept override
      {
        return {mWorkspaceSizes.data(), Parent::mDesc.getTargetCount()};
      }

    private:
      /**
       * @brief Implementation of the executeImpl method.
       * @param src View of the source data pointers.
       * @param dst View of the destination data pointers.
       * @param execParams The execution parameters.
       */
      void executeBackendImpl(View<void*> src, View<void*> dst, const afft::cuda::ExecutionParameters& execParams) override
      {
#     if CUFFT_VERSION < 10400
        if (execParams.stream != cudaStream_t{0})
        {
          throw Exception{Error::cufft, "until cuFFT 10.4, the default stream must be used"};
        }
#     endif
        checkError(cufftSetStream(mHandle, execParams.stream));

        // if (getDesc().useExternalWorkspace())
        // {
        //   checkError(cufftXtSetWorkArea(mHandle, execParams.externalWorkspaces.data()));
        // }

        std::copy(src.begin(), src.end(), mSrcDesc.data);
        std::copy(dst.begin(), dst.end(), mDstDesc.data);

        cudaLibXtDesc srcLibDesc{};
        srcLibDesc.descriptor = &mSrcDesc;
        srcLibDesc.library    = LIB_FORMAT_CUFFT;
        srcLibDesc.subFormat  = mSrcSubFormat;

        cudaLibXtDesc dstLibDesc{};
        dstLibDesc.descriptor = &mDstDesc;
        dstLibDesc.library    = LIB_FORMAT_CUFFT;
        dstLibDesc.subFormat  = mDstSubFormat;

        // Execute the transform
        checkError(cufftXtExecDescriptor(mHandle, &srcLibDesc, &dstLibDesc, getDirection()));        
      }

      Handle                                  mHandle{};         ///< The cuFFT plan handle.
      std::array<std::size_t, maxTargetCount> mWorkspaceSizes{}; ///< The size of the workspace
      std::array<std::size_t, maxTargetCount> mSrcElemCounts{};  ///< The number of elements in the source buffer
      std::array<std::size_t, maxTargetCount> mDstElemCounts{};  ///< The number of elements in the destination buffer
      cudaXtDesc                              mSrcDesc{};        ///< The source descriptor
      cudaXtDesc                              mDstDesc{};        ///< The destination descriptor
      int                                     mSrcSubFormat{};   ///< The source sub-format
      int                                     mDstSubFormat{};   ///< The destination sub-format
  };

  /**
   * @brief Create a cufft sp plan.
   * @param desc The plan descriptor.
   * @param backendParams The backend parameters.
   * @return The plan.
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE std::unique_ptr<afft::Plan>
  makePlan(const Description& desc, const afft::cuda::BackendParameters& backendParams)
  {
    const auto& descImpl = desc.get(DescToken::make());

    const auto cudaDevices   = descImpl.template getTargetDesc<Target::cuda>().getDevices();
    const auto precision     = descImpl.getPrecision().execution;
    const auto shape         = descImpl.getShape();
    const auto transformAxes = descImpl.getTransformAxes();
    const auto isRealDataDft = (descImpl.getTransformDesc<Transform::dft>().type != dft::Type::complexToComplex);

    if (const auto targetCount = desc.getTargetCount(); targetCount == 1)
    {
      const auto& memLayout = descImpl.template getMemDesc<MemoryLayout::centralized>();
      const auto  dftType   = descImpl.getTransformDesc<Transform::dft>().type;

      switch (precision)
      {
      case Precision::f16:
        // compute capability 5.3 or higher
        if (cuda::getComputeCapability(cudaDevices.front()) < cuda::ComputeCapability{5, 3})
        {
          throw Exception{Error::cufft, "f16 precision requires compute capability 5.3 or higher"};
        }

        // transformed dimensions must be powers of two
        if (std::any_of(transformAxes.begin(),
                        transformAxes.end(),
                        [&](auto axis){ return !isPowerOfTwo(shape[axis]); }))
        {
          throw Exception{Error::cufft, "transformed dimensions must be powers of two"};
        }

        // the fastest dimension must have unit stride for real part of the real data dft
        switch (dftType)
        {
        case dft::Type::realToComplex:
          if (memLayout.getSrcStrides()[transformAxes.back()] != 1)
          {
            throw Exception{Error::cufft, "the fastest dimension must have unit stride for real part of the real data dft"};
          }
          break;
        case dft::Type::complexToReal:
          if (memLayout.getDstStrides()[transformAxes.back()] != 1)
          {
            throw Exception{Error::cufft, "the fastest dimension must have unit stride for real part of the real data dft"};
          }
          break;
        default:
          break;
        }

        // the total number of elements must be less than 2^32
        if (std::accumulate(shape.begin(),
                            shape.end(),
                            Size{1},
                            std::multiplies<>{}) > std::numeric_limits<std::uint32_t>::max())
        {
          throw Exception{Error::cufft, "the total number of elements must be less than 2^32"};
        }

        break;
      case Precision::bf16:
        // compute capability 8.0 or higher
        if (cuda::getComputeCapability(cudaDevices.front()) < cuda::ComputeCapability{8, 0})
        {
          throw Exception{Error::cufft, "bf16 precision requires compute capability 8.0 or higher"};
        }

        // transformed dimensions must be powers of two
        if (std::any_of(transformAxes.begin(),
                        transformAxes.end(),
                        [&](auto axis){ return !isPowerOfTwo(shape[axis]); }))
        {
          throw Exception{Error::cufft, "transformed dimensions must be powers of two"};
        }

        // the fastest dimension must have unit stride for real part of the real data dft
        switch (dftType)
        {
        case dft::Type::realToComplex:
          if (memLayout.getSrcStrides()[transformAxes.back()] != 1)
          {
            throw Exception{Error::cufft, "the fastest dimension must have unit stride for real part of the real data dft"};
          }
          break;
        case dft::Type::complexToReal:
          if (memLayout.getDstStrides()[transformAxes.back()] != 1)
          {
            throw Exception{Error::cufft, "the fastest dimension must have unit stride for real part of the real data dft"};
          }
          break;
        default:
          break;
        }

        // the total number of elements must be less than 2^32
        if (std::accumulate(shape.begin(),
                            shape.end(),
                            Size{1},
                            std::multiplies<>{}) > std::numeric_limits<std::uint32_t>::max())
        {
          throw Exception{Error::cufft, "the total number of elements must be less than 2^32"};
        }

        break;
      case Precision::f32:
      case Precision::f64:
        break;
      default:
        throw Exception{Error::cufft, "unsupported precision"};
      }

#   if CUFFT_VERSION >= 11300
      if (precision != Precision::f32 && precision != Precision::f64)
      {
        throw Exception{Error::cufft, "normalization is supported only fo f32 and f64 precisions"};
      }
#   else
      if (descImpl.getNormalization() != Normalization::none)
      {
        throw Exception{Error::cufft, "normalization is not supported"};
      }
#   endif

      return std::make_unique<SingleDevicePlan>(desc, backendParams);
    }
    else if (targetCount > 1 && targetCount <= maxTargetCount)
    {
      // check if all devices have the same compute capability
      const auto ccRef = cuda::getComputeCapability(cudaDevices.front());
      if (std::any_of(std::next(cudaDevices.begin()),
                      cudaDevices.end(),
                      [&](auto device){ return cuda::getComputeCapability(device) != ccRef; }))
      {
        throw Exception{Error::cufft, "all devices must have the same compute capability"};
      }

      // check if all devices support UVA
      if (std::any_of(cudaDevices.begin(), cudaDevices.end(), [&](auto device){ return !cuda::hasUva(device); }))
      {
        throw Exception{Error::cufft, "all devices must support UVA"};
      }

      // check if each transformed dimension is greater than 32
      if (std::any_of(transformAxes.begin(), transformAxes.end(), [&](auto axis){ return shape[axis] < 32; }))
      {
        throw Exception{Error::cufft, "transformed dimensions must be less than 32"};
      }

      // check if the fastest dimension is even for real data ffts
      if (isRealDataDft && shape[transformAxes.back()] % 2 != 0)
      {
        throw Exception{Error::cufft, "the fastest dimension must be even for real data ffts"};
      }

      // check if the precision is f32 or f64
      switch (descImpl.getPrecision().execution)
      {
      case Precision::f32:
      case Precision::f64:
        break;
      default:
        throw Exception{Error::cufft, "unsupported precision"};
      }

      // check if the transform rank is 2 or 3
      if (const auto rank = descImpl.getTransformRank(); rank != 2 && rank != 3)
      {
        throw Exception{Error::cufft, "only 2D and 3D transforms are supported"};
      }

      if (const auto howManyRank = descImpl.getTransformHowManyRank(); howManyRank == 0)
      {
        if (descImpl.getPlacement() != Placement::inPlace)
        {
          throw Exception{Error::cufft, "non-batched transforms must be in-place"};
        }

        // TODO: check if src and dst distrib axes are the same
      }
      else if (howManyRank == 1)
      {
        if (descImpl.getTransformHowManyAxes().front() != 0)
        {
          throw Exception{Error::cufft, "only the first axis can be omitted"};
        }

        // TODO: check if src and dst distrib axes are the same
      }
      else
      {
        throw Exception{Error::cufft, "omitting more than one dimension is not supported"};
      }

      if (descImpl.getNormalization() != Normalization::none)
      {
        throw Exception{Error::cufft, "normalization is not supported"};
      }
      
      return std::make_unique<MultiDevicePlan>(desc, backendParams);
    }
    else
    {
      throw Exception(Error::cufft, "unsupported target count");
    }
  }
} // namespace afft::detail::cufft::sp

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_DETAIL_CUFFT_SP_HPP */
