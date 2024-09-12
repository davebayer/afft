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
  /**
   * @class PlanImpl
   * @brief Implementation of the sp plan interface for cuFFT.
   */
  class Plan final : public cufft::Plan<MpBackend::none>
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
      Plan(const Description& desc, const afft::cuda::BackendParameters& backendParams)
      : Parent{desc, backendParams}
      {        
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


        makeCufftPlan(mHandle, mDesc, mDesc, &mWorkspaceSize);
//         if (dftDesc.type == dft::Type::complexToComplex && std::all_of(n.begin(), n.end(), [](auto size){ return size <= 4096}))
//         {
// #       if CUFFT_VERSION >= 9200
//           checkError(cufftXtSetWorkAreaPolicy(planImpl->mHandle,
//                                               makeWorkAreaPolicy(cufftParams.workspacePolicy)),
//                                               &cufftParams.userWorkspaceSize);
// #       endif
//        }
      }

      /// @brief Destructor.
      ~Plan() override = default;

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
        //   checkError(cufftSetWorkArea(mHandle, execParams.workspace)); 
        // }

        checkError(cufftXtExec(mHandle, src.front(), dst.front(), getDirection()));
      }

      Handle      mHandle{};        ///< The cuFFT plan handle.
      std::size_t mWorkspaceSize{}; ///< The size of the workspace
      std::size_t mSrcElemCount{};  ///< The number of elements in the source buffer
      std::size_t mDstElemCount{};  ///< The number of elements in the destination buffer
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
    if (desc.getTargetCount() == 1)
    {
      const auto& descImpl = desc.get(DescToken::make());

      cuda::ScopedDevice scopedDevice{descImpl.getTargetDesc<Target::cuda>().getDevices()[0]};
      return std::make_unique<Plan>(desc, backendParams);
    }
    else
    {
      throw Exception(Error::cufft, "multi-GPU transforms are not yet supported");
    }
  }
} // namespace afft::detail::cufft::sp

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_DETAIL_CUFFT_SP_HPP */
