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

#ifndef AFFT_DETAIL_GPU_ROCFFT_PLAN_IMPL_HPP
#define AFFT_DETAIL_GPU_ROCFFT_PLAN_IMPL_HPP

#include <cstddef>
#include <memory>

#include <rocfft/rocfft.h>

#include "error.hpp"
#include "../../PlanImpl.hpp"

namespace afft::detail::gpu::rocfft
{
  /// @brief Rocfft size type
  using SizeT = std::size_t;

  /// @brief Rocfft plan deleter
  struct PlanDeleter
  {
    /**
     * @brief Destroy a rocFFT plan
     * @param plan rocFFT plan
     */
    void operator()(rocfft_plan plan) const
    {
      Error::check(rocfft_plan_destroy(plan));
    }
  };

  /// @brief Rocfft description deleter
  struct DescriptionDeleter
  {
    /**
     * @brief Destroy a rocFFT plan description
     * @param planDescription rocFFT plan description
     */
    void operator()(rocfft_plan_description planDescription) const
    {
      Error::check(rocfft_plan_description_destroy(planDescription));
    }
  };

  /// @brief Rocfft execution info deleter
  struct ExecInfoDeleter
  {
    /**
     * @brief Destroy a rocFFT execution info
     * @param info rocFFT execution info
     */
    void operator()(rocfft_execution_info info) const
    {
      Error::check(rocfft_execution_info_destroy(info));
    }
  };

  // struct HipMemDeleter
  // {
  //   void operator()(std::byte* ptr) const
  //   {
  //     Error::check(hipFree(ptr));
  //   }
  // };

  /// @brief RocFFT plan implementation
  class PlanImpl : public afft::detail::PlanImpl
  {
    private:
      /// @brief Alias to the parent class
      using Parent = afft::detail::PlanImpl;
    public:
      /// @brief Inherit constructors
      using Parent::Parent;

      /**
       * @brief Construct a new rocFFT plan
       * @param config Configuration
       */
      PlanImpl(const Config& config)
      : Parent(config)
      {
        const auto& dftParams    = getConfig().getTransformConfig<Transform::dft>();
        const auto& commonParams = getConfig().getCommonParameters();

        const auto rocfftPlacement = (commonParams.placement == Placement::inPlace)
                                       ? rocfft_placement_inplace : rocfft_placement_notinplace;

        rocfft_transform_type rocfftTransformType{};
        rocfft_array_type     rocfftInType{};
        rocfft_array_type     rocfftOutType{};
        switch (dftParams.type)
        {
          case dft::Type::complexToComplex:
            rocfftTransformType = (getConfig().getTransformDirection() == Direction::forward)
                                    ? rocfft_transform_type_complex_forward : rocfft_transform_type_complex_inverse;
            rocfftInType        = (commonParams.complexFormat == ComplexFormat::interleaved)
                                    ? rocfft_array_type_complex_interleaved : rocfft_array_type_complex_planar;
            rocfftOutType       = (commonParams.complexFormat == ComplexFormat::interleaved)
                                    ? rocfft_array_type_complex_interleaved : rocfft_array_type_complex_planar;
            break;
          case dft::Type::realToComplex:
            rocfftTransformType = rocfft_transform_type_real_forward;
            rocfftInType        = (commonParams.complexFormat == ComplexFormat::interleaved)
                                    ? rocfft_array_type_real : rocfft_array_type_real;
            rocfftOutType       = (commonParams.complexFormat == ComplexFormat::interleaved)
                                    ? rocfft_array_type_hermitian_interleaved : rocfft_array_type_hermitian_planar;
            break;
          case dft::Type::complexToReal:
            rocfftTransformType = rocfft_transform_type_real_inverse;
            rocfftInType        = (commonParams.complexFormat == ComplexFormat::interleaved)
                                    ? rocfft_array_type_hermitian_interleaved : rocfft_array_type_hermitian_planar;
            rocfftOutType       = (commonParams.complexFormat == ComplexFormat::interleaved)
                                    ? rocfft_array_type_real : rocfft_array_type_real;
            break;
          default:
            unreachable();
        }

        rocfft_precision rocfftPrecision{};
        switch (getConfig().getTransformPrecision().execution)
        {
        case Precision::f16: rocfftPrecision = rocfft_precision_half;   break;
        case Precision::f32: rocfftPrecision = rocfft_precision_single; break;
        case Precision::f64: rocfftPrecision = rocfft_precision_double; break;
        default:
          throw std::runtime_error("Unsupported precision");
        }

        const auto rocfftDimensions         = getConfig().getTransformRank();
        const auto rocfftLenghts            = getConfig().getTransformDims<SizeT>();
        const auto rocfftNumberOfTransforms = (getConfig().getTransformHowManyRank() > 0)
                                                ? getConfig().getTransformHowManyDims<SizeT>()[0]
                                                : SizeT{1};

        std::unique_ptr<std::remove_pointer_t<rocfft_plan_description>, DescriptionDeleter> rocfftDescription{};

        {
          rocfft_plan_description tmpDescription{};

          Error::check(rocfft_plan_description_create(&tmpDescription));

          rocfftDescription.reset(tmpDescription);
        }

        const auto rocfftInStrides  = getConfig().getTransformSrcStrides<SizeT>();
        const auto rocfftOutStrides = getConfig().getTransformDstStrides<SizeT>();
        const auto rocfftInDist     = (getConfig().getTransformHowManyRank() > 0)
                                        ? getConfig().getTransformHowManySrcStrides<SizeT>()[0] : SizeT{0};
        const auto rocfftOutDist    = (getConfig().getTransformHowManyRank() > 0)
                                        ? getConfig().getTransformHowManyDstStrides<SizeT>()[0] : SizeT{0};

        Error::check(rocfft_plan_description_set_data_layout(rocfftDescription.get(),
                                                             rocfftInType,
                                                             rocfftOutType,
                                                             nullptr,
                                                             nullptr,
                                                             rocfftDimensions,
                                                             rocfftInStrides.data(),
                                                             rocfftInDist,
                                                             rocfftDimensions,
                                                             rocfftOutStrides.data(),
                                                             rocfftOutDist));

        Error::check(rocfft_plan_description_set_scale_factor(rocfftDescription.get(), getConfig().getTransformNormFactor<Precision::f64>()));

        {
          rocfft_plan tmpPlan{};

          Error::check(rocfft_plan_create(&tmpPlan,
                                          rocfftPlacement,
                                          rocfftTransformType,
                                          rocfftPrecision,
                                          rocfftDimensions,
                                          rocfftLenghts.data(),
                                          rocfftNumberOfTransforms,
                                          rocfftDescription.get()));

          mPlan.reset(tmpPlan);
        }

        {
          rocfft_execution_info tmpExecInfo{};

          Error::check(rocfft_execution_info_create(&tmpExecInfo));

          mExecInfo.reset(tmpExecInfo);
        }

        Error::check(rocfft_plan_get_work_buffer_size(mPlan.get(), &mWorkspaceSize));

        // if (!getConfig().getTargetConfig<Target::gpu>().externalWorkspace && mWorkspaceSize > 0)
        // {
        //   {
        //     std::byte* tmpWorkspace{};

        //     Error::check(hipMalloc(&tmpWorkspace, mWorkspaceSize));

        //     mWorkspace.reset(tmpWorkspace);
        //   }

        //   Error::check(rocfft_execution_info_set_work_buffer(mExecInfo.get(), mWorkspace.get(), mWorkspaceSize));
        // }
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
        Error::check(rocfft_execution_info_set_stream(mExecInfo.get(), execParams.stream));

        if (getConfig().getTargetConfig<Target::gpu>().externalWorkspace)
        {
          Error::check(rocfft_execution_info_set_work_buffer(mExecInfo.get(), execParams.workspace, mWorkspaceSize));
        }

        Error::check(rocfft_execute(mPlan.get(), src.data(), dst.data(), mExecInfo.get()));
      }

      /**
       * @brief Get the workspace size
       * @return Workspace size
       */
      [[nodiscard]] std::size_t getWorkspaceSize() const noexcept override
      {
        return mWorkspaceSize;
      }
    protected:
    private:
      std::unique_ptr<std::remove_pointer_t<rocfft_plan>, PlanDeleter>               mPlan{};          ///< rocFFT plan
      std::unique_ptr<std::remove_pointer_t<rocfft_execution_info>, ExecInfoDeleter> mExecInfo{};      ///< rocFFT execution info
      // std::unique_ptr<std::byte, HipMemDeleter>                                      mWorkspace{};
      std::size_t                                                                    mWorkspaceSize{}; ///< Workspace size
  };

  /**
   * @brief Create a new rocFFT plan implementation
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
    case Precision::f16:
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
} // namespace afft::detail::gpu::rocfft

#endif /* AFFT_DETAIL_GPU_ROCFFT_PLAN_IMPL_HPP */
