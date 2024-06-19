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

#ifndef AFFT_DETAIL_ROCFFT_SPST_HPP
#define AFFT_DETAIL_ROCFFT_SPST_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../Plan.hpp"

#ifndef AFFT_DISABLE_GPU

namespace afft::detail::rocfft::spst::gpu
{
  /**
   * @brief Create a rocfft spst gpu plan implementation.
   * @param desc Plan description.
   * @return Plan implementation.
   */
  [[nodiscard]] std::unique_ptr<afft::Plan> makePlan(const Desc& desc);
} // namespace afft::detail::rocfft::spst::gpu

#ifdef AFFT_HEADER_ONLY

#include "Plan.hpp"

namespace afft::detail::rocfft::spst::gpu
{
  /**
   * @class Plan
   * @brief Implementation of the plan for the spst gpu architecture using rocFFT
   */
  class Plan final : public rocfft::Plan
  {
    private:
      /// @brief Alias for the parent class
      using Parent = rocfft::Plan;

    public:
      /// @brief inherit constructors
      using Parent::Parent;

      /**
       * @brief Constructor
       * @param Desc The plan description
       */
      Plan(const Desc& desc)
      : Parent{desc}
      {
        mDesc.fillDefaultMemoryLayoutStrides();

        hip::ScopedDevice device{mDesc.getArchDesc<Target::gpu, Distribution::spst>().device};

        {
          rocfft_plan plan{};
        
          checkError(rocfft_plan_create(&plan,
                                        getRocfftPlacement(),
                                        getRocfftTransformType(),
                                        getRocfftPrecision(),
                                        getRocfftDimensions(),
                                        getRocfftLengths().data(),
                                        getRocfftNumberOfTransforms(),
                                        getRocfftPlanDescription().get()));
          mPlan.reset(plan);
        }

        checkError(rocfft_plan_description_set_scale_factor(rocfftPlanDescription.get(),
                                                            mDesc.getNormalizationFactor<double>()));

        checkError(rocfft_plan_get_work_buffer_size(mPlan.get(), &mWorkspaceSize));

        {
          rocfft_execution_info info{};

          checkError(rocfft_execution_info_create(&info));

          mExecInfo.reset(info);
        }

        if (!mDesc.useExternalWorkspace())
        {
          hip::checkError(hipMalloc(&mWorkspace, mWorkspaceSize));

          checkError(rocfft_execution_info_set_work_buffer(mExecInfo.get(), mWorkspace, mWorkspaceSize));
        }
      }  

      /// @brief Destructor
      ~Plan() = default;

      /// @brief Inherit assignment operator
      using Parent::operator=;

      /**
       * @brief Execute the plan
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void executeBackendImpl(View<void*> src, View<void*> dst, const afft::spst::gpu::ExecutionParameters& execParams) override
      {
        checkError(rocfft_execution_info_set_stream(mExecInfo.get(), execParams.stream));

        if (mDesc.useExternalWorkspace())
        {
          checkError(rocfft_execution_info_set_work_buffer(mExecInfo.get(), execParams.workspace, mWorkspaceSize));
        }

        checkError(rocfft_execute(mPlan.get(), src.data(), dst.data(), mExecInfo.get()));
      }

      /**
       * @brief Get the workspace size
       * @return The workspace size
       */
      [[nodiscard]] constexpr View<std::size_t> getWorkspaceSize() const noexcept override
      {
        return {&mWorkspaceSize, 1};
      }
    protected:
    private:
      void*       mWorkspace{};     ///< The workspace
      std::size_t mWorkspaceSize{}; ///< The workspace size
  };

  /**
   * @brief Create a rocfft spst gpu plan implementation.
   * @param desc Plan description.
   * @return Plan implementation.
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE std::unique_ptr<afft::Plan> makePlan(const Desc& desc)
  {
    return std::make_unique<Plan>(desc);
  }
} // namespace afft::detail::rocfft::spst::gpu

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_DISABLE_GPU */

#endif /* AFFT_DETAIL_ROCFFT_SPST_HPP */
