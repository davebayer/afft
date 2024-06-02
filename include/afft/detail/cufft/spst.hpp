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

#ifndef AFFT_DETAIL_CUFFT_SPST_HPP
#define AFFT_DETAIL_CUFFT_SPST_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "common.hpp"
#include "error.hpp"
#include "../PlanImpl.hpp"

namespace afft::detail::cufft::spst
{
  /**
   * @class PlanImpl
   * @brief Implementation of the spst gpu plan interface for cuFFT.
   */
  class PlanImpl final : public detail::PlanImpl
  {
    private:
      /// @brief Alias for the parent class.
      using Parent = detail::PlanImpl;

    public:
      /**
       * @brief Factory method.
       * @param desc The plan descriptor.
       * @param initEffort The initialization effort.
       * @return The plan implementation.
       * @throw BackendError if an error occurs or the plan cannot be created.
       */
      [[nodiscard]] static std::unique_ptr<PlanImpl>
      make(const Desc& desc, const InitEffort initEffort)
      try
      {
        const auto& precision = desc.getTransformPrecision();
        const auto& dftDesc   = desc.getTransformDesc<Transform::dft>();
        const auto& gpuDesc   = desc.getTargetDesc<Target::gpu, Distribution::spst>();

        const auto rank = static_cast<int>(desc.getTransformRank());

        if (rank < 1 || rank > 3)
        {
          throw BackendError{Backend::cufft, "cuFFT supports only 1D, 2D, and 3D transforms"};
        }

        auto n                  = desc.getTransformDims<SizeT>();
        auto [inembed, istride] = desc.getTransformSrcNembedAndStride<SizeT>();
        auto idist              = (config.getTransformHowManyRank() == 1)
                                    ? desc.getTransformHowManySrcStrides<SizeT>().front()
                                    : SizeT{1};
        auto inputType          = makeCudaDatatype(precision.execution,
                                                   (dftParams.type == dft::Type::realToComplex)
                                                     ? Complexity::real : Complexity::complex);
        auto [onembed, ostride] = desc.getTransformDstNembedAndStride<SizeT>();
        auto odist              = (config.getTransformHowManyRank() == 1)
                                    ? desc.getTransformHowManyDstStrides<SizeT>().front()
                                    : SizeT{1};
        auto outputType         = makeCudaDatatype(precision.execution,
                                                   (dftParams.type == dft::Type::complexToReal)
                                                     ? Complexity::real : Complexity::complex);
        auto batch              = (config.getTransformHowManyRank() == 1)
                                    ? desc.getTransformHowManyDims<SizeT>().front() : SizeT{1};
        auto executionType      = makeCudaDatatype(precision.execution, Complexity::complex);

        cuda::ScopedDevice device{gpuDesc.device};

        auto planImpl = std::make_unique<PlanImpl>();

        if (initEffort >= InitEffort::med)
        {
#       if CUFFT_VERSION >= 11200
          checkError(cufftSetPlanPropertyInt64(planImpl->mHandle, NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT, 1));
#       endif
        }

        if (desc.useExternalWorkspace())
        {
          checkError(cufftSetAutoAllocation(planImpl->mHandle, 0));
        }

        std::size_t workspaceSize{};

        checkError(cufftXtMakePlanMany(planImpl->mHandle, rank, n.data(),
                                       inembed.data(), istride, idist, inputType,
                                       onembed.data(), ostride, odist, outputType,
                                       batch, &workspaceSize, executionType));

        if (dftDesc.type == dft::Type::complexToComplex && std::all_of(n.begin(), n.end(), [](auto size){ return size <= 4096}))
        {
#       if CUFFT_VERSION >= 9200
          checkError(cufftXtSetWorkAreaPolicy(planImpl->mHandle, makeWorkAreaPolicy(desc.getWorkspacePolicy())));
#       endif
        }

        return planImpl;
      }
      catch (const Exception&)
      {
        throw;
      }
      catch (const std::exception& e)
      {
        throw BackendError{Backend::cufft, e.what()};
      }

      /// @brief Deleted copy constructor.
      PlanImpl(const PlanImpl&) = delete;

      /// @brief Default move constructor.
      PlanImpl(PlanImpl&&) = default;

      /// @brief Deleted copy assignment operator.
      PlanImpl& operator=(const PlanImpl&) = delete;

      /// @brief Default move assignment operator.
      PlanImpl& operator=(PlanImpl&&) = default;

      /// @brief Destructor.
      ~PlanImpl() override
      {
        cufftDestroy(mHandle);
      }

      /**
       * @brief Implementation of the executeImpl method.
       * @param src View of the source data pointers.
       * @param dst View of the destination data pointers.
       * @param execParams The execution parameters.
       */
      void executeImpl(View<void*> src, View<void*> dst, const afft::gpu::spst::ExecutionParameters& execParams) override
      {
        if (src.size() != 1)
        {
          throw BackendError{Backend::cufft, "invalid number of source data pointers"};
        }

        if (dst.size() != 1)
        {
          throw BackendError{Backend::cufft, "invalid number of destination data pointers"};
        }

        const int direction = makeDirection(getDesc().getTransfomDirection());

        checkError(cufftSetStream(mHandle, execParams.stream));

        if (getDesc().useExternalWorkspace())
        {
          checkError(cufftSetWorkArea(mHandle, execParams.workspace)); 
        }

        checkError(cufftXtExec(mHandle, src.front(), dst.front(), direction));
      }
    private:
      /// @brief Constructor.
      PlanImpl(const Desc& desc)
      : Parent{desc}
      {
        checkError(cufftCreate(&mHandle));
      }

      cufftHandle mHandle{}; ///< The cuFFT plan handle.
  };
} // namespace afft::detail::cufft::spst

#endif /* AFFT_DETAIL_CUFFT_SPST_HPP */
