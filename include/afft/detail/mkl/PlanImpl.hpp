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

#ifndef AFFT_DETAIL_MKL_PLAN_IMPL_HPP
#define AFFT_DETAIL_MKL_PLAN_IMPL_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "error.hpp"
#include "../common.hpp"
#include "../cxx.hpp"
#include "../PlanImpl.hpp"

namespace afft::detail::mkl
{
  /// @brief Alias for MKL_LONG.
  using Long = MKL_LONG;

  /**
   * @brief Plan implementation.
   */
  class PlanImpl final : public detail::PlanImpl
  {
    private:
      /// @brief Alias for the parent class.
      using Parent = detail::PlanImpl;
    public:
      /// @brief Inherit constructors.
      using Parent::Parent;

      /**
       * @brief Constructor.
       * @param config Plan configuration.
       */
      PlanImpl(const Config& config)
      : Parent(config)
      {
        const auto& dftConfig    = getConfig().getTransformConfig<Transform::dft>();
        const auto& precision    = getConfig().getTransformPrecision();
        const auto& commonParams = getConfig().getCommonParameters();
        const auto direction     = getConfig().getTransformDirection();

        const auto rank = static_cast<Long>(getConfig().getTransformRank());
        const auto dims = getConfig().getTransformDims<Long>();

        const auto prec          = (precision.execution == Precision::f32) ? DFTI_SINGLE : DFTI_DOUBLE;
        const auto forwardDomain = (dftConfig.type != dft::Type::c2c) ? DFTI_REAL : DFTI_COMPLEX;

        {
          DFTI_DESCRIPTOR_HANDLE handle{};

          if (rank == 1)
          {
            Error::check(DftiCreateDescriptor(&handle, prec, forwardDomain, 1, dims[0]));
          }
          else
          {
            Error::check(DftiCreateDescriptor(&handle, prec, forwardDomain, rank, dims.data()));
          }

          if (handle == nullptr)
          {
            throw makeException<std::runtime_error>("Failed to create descriptor");
          }

          mHandle.reset(handle);
        }

        const auto placement = (commonParams.placement == Placement::inPlace) ? DFTI_INPLACE : DFTI_NOT_INPLACE;
        Error::check(DftiSetValue(mHandle.get(), DFTI_PLACEMENT, placement));

        const auto scaleConfigParam = (direction == Direction::forward) ? DFTI_FORWARD_SCALE : DFTI_BACKWARD_SCALE;
        const auto scale            = getConfig().getTransformNormFactor<Precision::f64>();
        Error::check(DftiSetValue(mHandle.get(), scaleConfigParam, scale));

        const auto& cpuConfig = getConfig().getTargetConfig<Target::cpu>();
        Error::check(DftiSetValue(mHandle.get(), DFTI_THREAD_LIMIT, static_cast<MKL_LONG>(cpuConfig.threadLimit)));

        std::array<Long, maxDimCount + 1> strides{};
        const auto srcStrides = getConfig().getTransformSrcStrides<Long>();
        std::copy(srcStrides.begin(), srcStrides.end(), std::next(strides.begin()));
        Error::check(DftiSetValue(mHandle.get(), DFTI_INPUT_STRIDES, strides.data()));

        const auto dstStrides = getConfig().getTransformDstStrides<Long>();
        std::copy(dstStrides.begin(), dstStrides.end(), std::next(strides.begin()));
        Error::check(DftiSetValue(mHandle.get(), DFTI_OUTPUT_STRIDES, strides.data()));

        //FIXME: For in-place transforms (DFTI_PLACEMENT=DFTI_INPLACE), the configuration set by DFTI_OUTPUT_STRIDES is ignored when the element types in the forward and backward domains are the same.

        if (const auto howManyRank = getConfig().getTransformHowManyRank(); howManyRank > 0)
        {
          const auto howManyDims       = getConfig().getTransformHowManyDims<Long>();
          const auto howManySrcStrides = getConfig().getTransformHowManySrcStrides<Long>();
          const auto howManyDstStrides = getConfig().getTransformHowManyDstStrides<Long>();

          Error::check(DftiSetValue(mHandle.get(), DFTI_NUMBER_OF_TRANSFORMS, howManyDims[howManyRank - 1]));
          Error::check(DftiSetValue(mHandle.get(), DFTI_INPUT_DISTANCE, howManySrcStrides[howManyRank - 1]));
          Error::check(DftiSetValue(mHandle.get(), DFTI_OUTPUT_DISTANCE, howManyDstStrides[howManyRank - 1]));
        }

        switch (dftConfig.type)
        {
        case dft::Type::complexToComplex:
          if (commonParams.complexFormat == ComplexFormat::interleaved)
          {
            Error::check(DftiSetValue(mHandle.get(), DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX));
          }
          else
          {
            Error::check(DftiSetValue(mHandle.get(), DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL));
          }
          break;
        case dft::Type::realToComplex:
        case dft::Type::complexToReal:
          if (commonParams.complexFormat == ComplexFormat::planar)
          {
            throw makeException<std::runtime_error>("Planar format is not supported for real-to-complex or complex-to-real transforms");
          }

          Error::check(DftiSetValue(mHandle.get(), DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
          break;
        default:
          cxx::unreachable();
        }

        const auto workspace = (commonParams.workspacePolicy != WorkspacePolicy::minimal)
                                 ? DFTI_ALLOW : DFTI_AVOID;
        Error::check(DftiSetValue(mHandle.get(), DFTI_WORKSPACE, workspace));

        const auto destroySource = (commonParams.destroySource) ? DFTI_ALLOW : DFTI_AVOID;
        Error::check(DftiSetValue(mHandle.get(), DFTI_DESTROY_INPUT, destroySource));

        Error::check(DftiCommitDescriptor(mHandle.get()));
      }

      /**
       * @brief Execute the plan implementation.
       * @param src Source data.
       * @param dst Destination data.
       * @param params Execution parameters.
       */
      void executeImpl(ExecParam src, ExecParam dst, const afft::cpu::ExecutionParameters&) override
      {
        const auto computeFn = (getConfig().getTransformDirection() == Direction::forward)
                                 ? DftiComputeForward : DftiComputeBackward;
        const auto placement = getConfig().getCommonParameters().placement;

        if (getConfig().getCommonParameters().complexFormat == ComplexFormat::interleaved)
        {
          if (placement == Placement::inPlace)
          {
            Error::check(computeFn(mHandle.get(), src.getRealImag()));
          }
          else
          {
            Error::check(computeFn(mHandle.get(), src.getRealImag(), dst.getRealImag()));
          }
        }
        else
        {
          if (placement == Placement::inPlace)
          {
            Error::check(computeFn(mHandle.get(), src.getReal(), src.getImag()));
          }
          else
          {
            Error::check(computeFn(mHandle.get(), src.getReal(), src.getImag(), dst.getReal(), dst.getImag()));
          }
        }
      }
    protected:
    private:
      /**
       * @brief Deleter for DFTI descriptor handle.
       */
      struct Deleter
      {
        /**
         * @brief Delete DFTI descriptor handle.
         * @param handle DFTI descriptor handle.
         */
        void operator()(DFTI_DESCRIPTOR_HANDLE handle) const
        {
          if (handle != nullptr)
          {
            DftiFreeDescriptor(&handle);
          }
        }
      };

      std::unique_ptr<std::remove_pointer_t<DFTI_DESCRIPTOR_HANDLE>, Deleter> mHandle{}; ///< MKL DFTI descriptor handle.
  };

  /**
   * @brief Create a plan implementation.
   * @param config Plan configuration.
   * @return Plan implementation.
   */
  [[nodiscard]] std::unique_ptr<detail::PlanImpl> makePlanImpl(const Config& config)
  {
    switch (config.getTransform())
    {
    case Transform::dft:
      break;
    default:
      throw makeException<std::runtime_error>("Unsupported transform");
    }

    if (config.getTransformHowManyRank() > 1)
    {
      throw makeException<std::runtime_error>("Unsupported howMany rank");
    }

    const auto& precision = config.getTransformPrecision();

    if (precision.source != precision.execution || precision.destination != precision.execution)
    {
      throw makeException<std::runtime_error>("All precisions must be the same for CPU backend");
    }

    switch (precision.execution)
    {
    case Precision::f32: case Precision::f64:
      break;    
    default:
      throw makeException<std::runtime_error>("Unsupported precision");
    }

    return std::make_unique<PlanImpl>(config);
  }
} // namespace afft::detail::mkl

#endif /* AFFT_DETAIL_MKL_PLAN_IMPL_HPP */
