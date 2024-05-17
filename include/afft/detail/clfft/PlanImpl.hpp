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

#ifndef AFFT_DETAIL_CLFFT_PLAN_IMPL_HPP
#define AFFT_DETAIL_CLFFT_PLAN_IMPL_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "error.hpp"
#include "../common.hpp"
#include "../PlanImpl.hpp"

namespace afft::detail::clfft
{
  /// @brief clFFT size type
  using SizeT = std::size_t;

  /// @brief clFFT plan implementation
  class PlanImpl : public afft::detail::PlanImpl
  {
    private:
      /// @brief Alias to the parent class
      using Parent = afft::detail::PlanImpl;
    public:
      /// @brief Inherit constructors
      using Parent::Parent;

      /**
       * @brief Construct a new clFFT plan
       * @param config Configuration
       */
      PlanImpl(const Config& config)
      : Parent(config)
      {
        const auto& gpuConfig    = getConfig().getTargetConfig<Target::gpu>();
        const auto& dftParams    = getConfig().getTransformConfig<Transform::dft>();
        const auto& commonParams = getConfig().getCommonParameters();

        const auto clfftDirection = (getConfig().getTransformDirection() == Direction::forward)
                                      ? CLFFT_FORWARD : CLFFT_BACKWARD;

        clfftDim clfftDim{};
        switch (getConfig().getTransformRank())
        {
        case 1: clfftDim = CLFFT_1D; break;
        case 2: clfftDim = CLFFT_2D; break;
        case 3: clfftDim = CLFFT_3D; break;
        default:
          throw std::runtime_error("Unsupported rank");
        }

        const auto clfftLengths = getConfig().getTransformDims<SizeT>();

        {
          clfftPlanHandle planHandle{};

          Error::check(clfftCreateDefaultPlan(&planHandle,
                                              gpuConfig.context,
                                              clfftDim,
                                              clfftLengths.data()));

          mPlanHandle = planHandle;
        }

        clfftPrecision clfftPrecision{};
        switch (getConfig().getTransformPrecision().execution)
        {
        case Precision::f32: clfftPrecision = CLFFT_SINGLE; break;
        case Precision::f64: clfftPrecision = CLFFT_DOUBLE; break;
        default:
          throw std::runtime_error("Unsupported precision");
        }

        Error::check(clfftSetPlanPrecision(mPlanHandle.value(), clfftPrecision));

        const auto clfftScale = getConfig().getTransformNormFactor<Precision::f32>();

        Error::check(clfftSetPlanScale(mPlanHandle.value(), clfftDirection, clfftScale));

        const auto clfftBatchSize = (getConfig().getTransformHowManyRank() > 0)
                                      ? getConfig().getTransformHowManyDims<SizeT>()[0] : SizeT{1};

        Error::check(clfftSetPlanBatchSize(mPlanHandle.value(), clfftBatchSize));

        auto clfftInStride  = getConfig().getTransformSrcStrides<SizeT>();
        auto clfftOutStride = getConfig().getTransformDstStrides<SizeT>();

        Error::check(clfftSetPlanInStride(mPlanHandle.value(), clfftDim, clfftInStride.data()));
        Error::check(clfftSetPlanOutStride(mPlanHandle.value(), clfftDim, clfftOutStride.data()));

        if (getConfig().getTransformHowManyRank() > 0)
        {
          auto clfftInDist  = getConfig().getTransformHowManySrcStrides<SizeT>()[0];
          auto clfftOutDist = getConfig().getTransformHowManyDstStrides<SizeT>()[0];

          Error::check(clfftSetPlanDistance(mPlanHandle.value(), clfftInDist, clfftOutDist));
        }

        clfftLayout clfftInLayout{};
        clfftLayout clfftOutLayout{};
        switch (dftParams.type)
        {
        case dft::Type::complexToComplex:
          clfftInLayout  = (commonParams.complexFormat == ComplexFormat::interleaved)
                             ? CLFFT_COMPLEX_INTERLEAVED : CLFFT_COMPLEX_PLANAR;
          clfftOutLayout = (commonParams.complexFormat == ComplexFormat::interleaved)
                             ? CLFFT_COMPLEX_INTERLEAVED : CLFFT_COMPLEX_PLANAR;
          break;
        case dft::Type::realToComplex:
          clfftInLayout  = (commonParams.complexFormat == ComplexFormat::interleaved)
                             ? CLFFT_REAL : CLFFT_REAL;
          clfftOutLayout = (commonParams.complexFormat == ComplexFormat::interleaved)
                             ? CLFFT_HERMITIAN_INTERLEAVED : CLFFT_HERMITIAN_PLANAR;
          break;
        case dft::Type::complexToReal:
          clfftInLayout  = (commonParams.complexFormat == ComplexFormat::interleaved)
                             ? CLFFT_HERMITIAN_INTERLEAVED : CLFFT_HERMITIAN_PLANAR;
          clfftOutLayout = (commonParams.complexFormat == ComplexFormat::interleaved)
                             ? CLFFT_REAL : CLFFT_REAL;
          break;
        default:
          cxx::unreachable();
        }

        Error::check(clfftSetLayout(mPlanHandle.value(), clfftInLayout, clfftOutLayout));

        const auto clfftResultLocation = (commonParams.placement == Placement::inPlace)
                                           ? CLFFT_INPLACE : CLFFT_OUTOFPLACE;

        Error::check(clfftSetResultLocation(mPlanHandle.value(), clfftResultLocation));

        Error::check(clfftBakePlan(mPlanHandle.value(), 0, nullptr, nullptr, nullptr));
      }

      /// @brief Destructor
      ~PlanImpl() override
      {
        if (mPlanHandle)
        {
          Error::check(clfftDestroyPlan(&mPlanHandle.value()));
        }
      }

      /// @brief Inherit assignment operator
      using Parent::operator=;

      /**
       * @brief Execute the plan
       * @param src Source data
       * @param dst Destination data
       * @param execParams Execution parameters
       */
      void executeImpl(ExecParam src, ExecParam dst, const afft::gpu::ExecutionParameters& execParams) override
      {
        const auto clfftDirection = (getConfig().getTransformDirection() == Direction::forward)
                                      ? CLFFT_FORWARD : CLFFT_BACKWARD;

        auto tmpBuffer = (getConfig().getTargetConfig<Target::gpu>().externalWorkspace)
                           ? execParams.workspace : nullptr;

        Error::check(clfftEnqueueTransform(mPlanHandle.value(),
                                           clfftDirection,
                                           1,
                                           &execParams.commandQueue,
                                           0,
                                           nullptr,
                                           nullptr,
                                           reinterpret_cast<cl_mem*>(src.data()),
                                           reinterpret_cast<cl_mem*>(dst.data()),
                                           tmpBuffer));
      }

      /**
       * @brief Get the workspace size
       * @return Workspace size
       */
      [[nodiscard]] std::size_t getWorkspaceSize() const override
      {
        std::size_t workspaceSize{};

        Error::check(clfftGetTmpBufSize(mPlanHandle.value(), &workspaceSize));

        return workspaceSize;
      }
    protected:
    private:
      std::optional<clfftPlanHandle> mPlanHandle{};
  };

  /**
   * @brief Create a new clFFT plan implementation
   * @param config Configuration
   * @return Plan implementation
   */
  [[nodiscard]] inline std::unique_ptr<PlanImpl> makePlanImpl(const Config& config)
  {
    switch (config.getTransform())
    {
    case Transform::dft:
      break;    
    default:
      throw std::runtime_error("Unsupported transform");
    }

    if (config.getTransformRank() > 3)
    {
      throw std::runtime_error("Rank must be less than or equal to 3");
    }

    if (config.getTransformHowManyRank() > 1)
    {
      throw std::runtime_error("How many rank must be less than or equal to 1");
    }

    const auto& prec = config.getTransformPrecision();
    if (prec.source != prec.execution || prec.source != prec.destination)
    {
      throw std::runtime_error("Source, execution and destination precision must be the same");
    }

    switch (prec.execution)
    {
    case Precision::f32:
    case Precision::f64:
      break;
    default:
      throw std::runtime_error("Unsupported precision");
    }

    const auto& commonParams = config.getCommonParameters();
    const auto& dftConfig    = config.getTransformConfig<Transform::dft>();

    switch (dftConfig.type)
    {
    case dft::Type::realToComplex:
    case dft::Type::complexToReal:
      if (commonParams.complexFormat == ComplexFormat::planar && commonParams.placement == Placement::inPlace)
      {
        throw std::runtime_error("Planar format is not supported for real in-place transforms");
      }
      break;
    default:
      break;
    }

    return std::make_unique<PlanImpl>(config);
  }
} // namespace afft::detail::clfft

#endif /* AFFT_DETAIL_CLFFT_PLAN_IMPL_HPP */
