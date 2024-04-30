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

#ifndef AFFT_DETAIL_CONFIG_HPP
#define AFFT_DETAIL_CONFIG_HPP

#include <cstddef>
#include <cmath>
#include <functional>
#include <stdexcept>

#include "common.hpp"
#include "DimensionsConfig.hpp"
#include "TargetConfig.hpp"
#include "TransformConfig.hpp"
#include "utils.hpp"

namespace afft::detail
{
  /**
   * @class Config
   * @brief Configuration for the transform.
   */
  class Config
  {
    public:
      /// @brief The default constructor.
      Config() = delete;

      /**
       * @brief The constructor.
       * @param transformParams The parameters for the transform.
       * @param targetParams The parameters for the target platform.
       */
      explicit constexpr
      Config(const TransformParametersType auto& transformParams, const TargetParametersType auto& targetParams)
      : mCommonParams(checkCommonParameters(transformParams.commonParameters)),
        mDimsConfig(transformParams.dimensions),
        mTransformConfig(transformParams),
        mTargetConfig(targetParams)
      {
        // correct strides in dimensions config
        mTransformConfig.correctDimensionsConfig(mDimsConfig, getCommonParameters());
      }

      /// @brief The copy constructor.
      Config(const Config&) = default;

      /// @brief The move constructor.
      Config(Config&&) = default;

      /// @brief The destructor.
      ~Config() = default;

      /// @brief The copy assignment operator.
      Config& operator=(const Config&) = default;

      /// @brief The move assignment operator.
      Config& operator=(Config&&) = default;

      /**
       * @brief Get common parameters.
       * @return The common parameters.
       */
      [[nodiscard]] constexpr const CommonParameters& getCommonParameters() const noexcept
      {
        return mCommonParams;
      }

      /**
       * @brief Get shape rank.
       * @return The shape rank.
       */
      [[nodiscard]] constexpr std::size_t getShapeRank() const noexcept
      {
        return mDimsConfig.getRank();
      }

      /**
       * @brief Get shape.
       * @return The shape.
       */
      [[nodiscard]] constexpr std::span<const std::size_t> getShape() const noexcept
      {
        return mDimsConfig.getShape();
      }

      /**
       * @brief Get source strides.
       * @return The source strides.
       */
      [[nodiscard]] constexpr std::span<const std::size_t> getSrcStrides() const noexcept
      {
        return mDimsConfig.getSrcStrides();
      }

      /**
       * @brief Get destination strides.
       * @return The destination strides.
       */
      [[nodiscard]] constexpr std::span<const std::size_t> getDstStrides() const noexcept
      {
        return mDimsConfig.getDstStrides();
      }

      /**
       * @brief Are source and destination strides equal?
       * @return True if equal, false otherwise.
       */
      [[nodiscard]] constexpr bool hasEqualStrides() const noexcept
      {
        return mDimsConfig.stridesEqual();
      }

      /**
       * @brief Get transform direction.
       * @return The transform direction.
       */
      [[nodiscard]] constexpr Direction getTransformDirection() const noexcept
      {
        return mTransformConfig.getDirection();
      }

      /**
       * @brief Get transform precision.
       * @return The transform precision.
       */
      [[nodiscard]] constexpr const PrecisionTriad& getTransformPrecision() const noexcept
      {
        return mTransformConfig.getPrecision();
      }

      /**
       * @brief Get transform rank.
       * @return The transform rank.
       */
      [[nodiscard]] constexpr std::size_t getTransformRank() const noexcept
      {
        return mTransformConfig.getRank();
      }

      /**
       * @brief Get transform axes.
       * @return The transform axes.
       */
      [[nodiscard]] constexpr std::span<const std::size_t> getTransformAxes() const noexcept
      {
        return mTransformConfig.getAxes();
      }

      /**
       * @brief Get transform type.
       * @return The transform type.
       */
      [[nodiscard]] constexpr Transform getTransform() const noexcept
      {
        return mTransformConfig.getType();
      }

      /**
       * @brief Get transform configuration.
       * @return The transform configuration.
       */
      template<Transform transform>
      [[nodiscard]] constexpr const auto& getTransformConfig() const
      {
        return mTransformConfig.getConfig<transform>();
      }

      /**
       * @brief Get transform parameters.
       * @tparam transform The transform.
       * @return The transform parameters.
       */
      template<Transform transform>
      [[nodiscard]] constexpr TransformParameters<transform> getTransformParameters() const
      {
        Dimensions dims{.shape     = mDimsConfig.getShape(),
                        .srcStride = mDimsConfig.getSrcStrides(),
                        .dstStride = mDimsConfig.getDstStrides()};

        if constexpr (transform == Transform::dft)
        {
          return dft::Parameters{.dimensions       = std::move(dims),
                                 .commonParameters = mCommonParams,
                                 .axes             = mTransformConfig.getAxes(),
                                 .direction        = mTransformConfig.getDirection(),
                                 .precision        = mTransformConfig.getPrecision(),
                                 .type             = getTransformConfig<Transform::dft>().type};
        }
        else if constexpr (transform == Transform::dtt)
        {
          return dtt::Parameters{.dimensions       = std::move(dims),
                                 .commonParameters = mCommonParams,
                                 .direction        = mTransformConfig.getDirection(),
                                 .precision        = mTransformConfig.getPrecision(),
                                 .axes             = mTransformConfig.getAxes(),
                                 .types            = getTransformConfig<Transform::dtt>().axisTypes};
        }
        else
        {
          unreachable();
        }
      }

      /**
       * @brief Get normalization factor.
       * @tparam prec The precision.
       * @return The normalization factor.
       */
      template<Precision prec>
      [[nodiscard]] constexpr auto getTransformNormFactor() const
      {
        Real<prec> factor{1};

        const auto logSize = mTransformConfig.getTransformLogicalSize(mDimsConfig.getShape());

        switch (mCommonParams.normalize)
        {
          case Normalize::none:
            break;
          case Normalize::orthogonal:
            factor /= std::sqrt(static_cast<Real<prec>>(logSize));
            break;
          case Normalize::unitary:
            factor /= static_cast<Real<prec>>(logSize);
            break;
          default:
            unreachable();
        }

        return factor;
      }

      /**
       * @brief Get transform dimensions.
       * @tparam T The type of the returned dimension size elements. Must be integral.
       * @return The transform dimensions.
       */
      template<std::integral I>
      [[nodiscard]] constexpr MaxDimArray<I> getTransformDims() const
      {
        MaxDimArray<I> dims{};

        for (std::size_t i{}; i < mTransformConfig.getRank(); ++i)
        {
          dims[i] = safeIntCast<I>(mDimsConfig.getShape()[mTransformConfig.getAxes()[i]]);
        }

        return dims;
      }

      /**
       * @brief Get transform source strides.
       * @tparam T The type of the returned stride size elements. Must be integral.
       * @return The transform strides.
       */
      template<std::integral I>
      [[nodiscard]] constexpr MaxDimArray<I> getTransformSrcStrides() const
      {
        MaxDimArray<I> strides{};

        for (std::size_t i{}; i < mTransformConfig.getRank(); ++i)
        {
          strides[i] = safeIntCast<I>(mDimsConfig.getSrcStrides()[mTransformConfig.getAxes()[i]]);
        }

        return strides;
      }

      /**
       * @brief Get transform destination strides.
       * @tparam T The type of the returned stride size elements. Must be integral.
       * @return The transform strides.
       */
      template<std::integral I>
      [[nodiscard]] constexpr MaxDimArray<I> getTransformDstStrides() const
      {
        MaxDimArray<I> strides{};

        for (std::size_t i{}; i < mTransformConfig.getRank(); ++i)
        {
          strides[i] = safeIntCast<I>(mDimsConfig.getDstStrides()[mTransformConfig.getAxes()[i]]);
        }

        return strides;
      }

      /**
       * @brief Get transform source nembed and stride.
       * @tparam T The type of the returned nembed and stride size elements. Must be integral.
       * @return The transform nembed and stride.
       */
      template<std::integral I>
      [[nodiscard]] constexpr auto getTransformSrcNembedAndStride() const
      {
        auto getStride = [this](std::size_t i)
        {
          return mDimsConfig.getSrcStrides()[mTransformConfig.getAxes()[i]];
        };

        MaxDimArray<I> nembed{};
        const I stride = safeIntCast<I>(getStride(mTransformConfig.getRank() - 1));

        if (mTransformConfig.getRank() == 1)
        {
          nembed[0] = safeIntCast<I>(mDimsConfig.getShape()[mTransformConfig.getAxes()[0]]);
        }

        for (std::size_t i{}; i < mTransformConfig.getRank() - 1; ++i)
        {
          auto [n, rem] = div(getStride(i), getStride(i + 1));

          if (rem != 0)
          {
            throw makeException<std::runtime_error>("Passed stride parameters do not support nembed-like transformation");
          }

          if (n < mDimsConfig.getShape()[mTransformConfig.getAxes()[i]])
          {
            throw makeException<std::runtime_error>("Passed stride parameters do not support nembed-like transformation");
          }

          nembed[i] = safeIntCast<I>(n);
        }

        return std::make_tuple(nembed, stride);
      }

      /**
       * @brief Get transform destination nembed and stride.
       * @tparam T The type of the returned nembed and stride size elements. Must be integral.
       * @return The transform nembed and stride.
       */
      template<std::integral I>
      [[nodiscard]] constexpr auto getTransformDstNembedAndStride() const
      {
        auto getStride = [this](std::size_t i)
        {
          return mDimsConfig.getDstStrides()[mTransformConfig.getAxes()[i]];
        };

        MaxDimArray<I> nembed{};
        const I stride = safeIntCast<I>(getStride(mTransformConfig.getRank() - 1));

        if (mTransformConfig.getRank() == 1)
        {
          nembed[0] = safeIntCast<I>(mDimsConfig.getShape()[mTransformConfig.getAxes()[0]]);
        }

        for (std::size_t i{}; i < mTransformConfig.getRank() - 1; ++i)
        {
          auto [n, rem] = div(getStride(i), getStride(i + 1));

          if (rem != 0)
          {
            throw makeException<std::runtime_error>("Passed stride parameters do not support nembed-like transformation");
          }

          if (n < mDimsConfig.getShape()[mTransformConfig.getAxes()[i]])
          {
            throw makeException<std::runtime_error>("Passed stride parameters do not support nembed-like transformation");
          }

          nembed[i] = safeIntCast<I>(n);
        }

        return std::make_tuple(nembed, stride);
      }

      /**
       * @brief Get transform how many rank.
       * @return The transform how many rank.
       */
      [[nodiscard]] constexpr std::size_t getTransformHowManyRank() const noexcept
      {
        return mDimsConfig.getRank() - mTransformConfig.getRank();
      }

      /**
       * @brief Get transform how many dimensions.
       * @tparam T The type of the returned dimension size elements. Must be integral.
       * @return The transform how many dimensions.
       */
      template<std::integral I>
      [[nodiscard]] constexpr MaxDimArray<I> getTransformHowManyDims() const
      {
        MaxDimArray<I> dims{};

        const auto axes = mTransformConfig.getAxes();

        for (std::size_t i{}, pos{}; i < mDimsConfig.getRank(); ++i)
        {
          if (std::find(axes.begin(), axes.end(), i) == axes.end())
          {
            dims[pos++] = safeIntCast<I>(mDimsConfig.getShape()[i]);
          }
        }

        return dims;
      }

      /**
       * @brief Get transform how many source strides.
       * @tparam T The type of the returned stride size elements. Must be integral.
       * @return The transform how many strides.
       */
      template<std::integral I>
      [[nodiscard]] constexpr MaxDimArray<I> getTransformHowManySrcStrides() const
      {
        MaxDimArray<I> strides{};

        const auto axes = mTransformConfig.getAxes();

        for (std::size_t i{}, pos{}; i < mDimsConfig.getRank(); ++i)
        {
          if (std::find(axes.begin(), axes.end(), i) == axes.end())
          {
            strides[pos++] = safeIntCast<I>(mDimsConfig.getSrcStrides()[i]);
          }
        }

        return strides;
      }

      /**
       * @brief Get transform how many destination strides.
       * @tparam T The type of the returned stride size elements. Must be integral.
       * @return The transform how many strides.
       */
      template<std::integral I>
      [[nodiscard]] constexpr MaxDimArray<I> getTransformHowManyDstStrides() const
      {
        MaxDimArray<I> strides{};

        const auto axes = mTransformConfig.getAxes();

        for (std::size_t i{}, pos{}; i < mDimsConfig.getRank(); ++i)
        {
          if (std::find(axes.begin(), axes.end(), i) == axes.end())
          {
            strides[pos++] = safeIntCast<I>(mDimsConfig.getDstStrides()[i]);
          }
        }

        return strides;
      }

      /**
       * @brief Get size of source element in bytes.
       * @return The size of source element in bytes.
       */
      [[nodiscard]] constexpr std::size_t sizeOfSrcElem() const
      {
        const auto precision = getTransformPrecision();

        std::size_t srcElemSizeOf = sizeOf(precision.source);

        if (mCommonParams.complexFormat == ComplexFormat::interleaved)
        {
          srcElemSizeOf *= 2;
        }

        return srcElemSizeOf;
      }

      /**
       * @brief Get size of destination element in bytes.
       * @return The size of destination element in bytes.
       */
      [[nodiscard]] constexpr std::size_t sizeOfDstElem() const
      {
        const auto precision = getTransformPrecision();

        std::size_t dstElemSizeOf = sizeOf(precision.destination);

        if (mCommonParams.complexFormat == ComplexFormat::interleaved)
        {
          dstElemSizeOf *= 2;
        }

        return dstElemSizeOf;
      }

      /**
       * @brief Get volume of source shape.
       * @return The volume of source shape in terms of elements.
       */
      [[nodiscard]] constexpr std::size_t getSrcShapeVolume() const noexcept
      {
        const auto shape      = mDimsConfig.getShape();
        const auto srcStrides = mDimsConfig.getSrcStrides();

        std::size_t size = 1;

        for (std::size_t i{}; i < mDimsConfig.getRank(); ++i)
        {
          size *= shape[i] * srcStrides[i];
        }

        return size;
      }


      /**
       * @brief Get volume of destination shape.
       * @return The volume of destination shape in terms of elements.
       */
      [[nodiscard]] constexpr std::size_t getDstShapeVolume() const noexcept
      {
        const auto shape      = mDimsConfig.getShape();
        const auto dstStrides = mDimsConfig.getDstStrides();

        std::size_t size = 1;

        for (std::size_t i{}; i < mDimsConfig.getRank(); ++i)
        {
          size *= shape[i] * dstStrides[i];
        }

        return size;
      }

      /**
       * @brief Get source complexity.
       * @return The source complexity.
       */
      [[nodiscard]] constexpr Complexity getSrcComplexity() const noexcept
      {
        return getSrcDstComplexity().first;
      }

      /**
       * @brief Get destination complexity.
       * @return The destination complexity.
       */
      [[nodiscard]] constexpr Complexity getDstComplexity() const noexcept
      {
        return getSrcDstComplexity().second;
      }

      /**
       * @brief Get complexity of source and destination.
       * @return The complexity of source and destination.
       */
      [[nodiscard]] constexpr std::pair<Complexity, Complexity> getSrcDstComplexity() const noexcept
      {
        Complexity srcComplexity{};
        Complexity dstComplexity{};

        switch (getTransform())
        {
        case Transform::dft:
        {
          const auto& dftConfig = getTransformConfig<Transform::dft>();

          srcComplexity = (dftConfig.type != dft::Type::realToComplex) ? Complexity::complex : Complexity::real;
          dstComplexity = (dftConfig.type != dft::Type::complexToReal) ? Complexity::complex : Complexity::real;
          break;
        }
        case Transform::dtt:
          srcComplexity = Complexity::real;
          dstComplexity = Complexity::real;
          break;
        default:
          break;
        }

        return std::make_pair(srcComplexity, dstComplexity);
      }

      /**
       * @brief Get target.
       * @return The target.
       */
      [[nodiscard]] constexpr Target getTarget() const noexcept
      {
        return mTargetConfig.getTarget();
      }

      /**
       * @brief Get target configuration.
       * @tparam target The target.
       * @return The target configuration.
       */
      template<Target target>
      [[nodiscard]] constexpr const auto& getTargetConfig() const
      {
        return mTargetConfig.getConfig<target>();
      }

      /**
       * @brief Get target parameters.
       * @tparam target The target.
       * @return The target parameters.
       */
      template<Target target>
      [[nodiscard]] constexpr TargetParameters<target> getTargetParameters() const
      {
        if constexpr (target == Target::cpu)
        {
          const auto& cpuParams = getTargetConfig<Target::cpu>();

          return afft::cpu::Parameters{.alignment = cpuParams.alignment, .threadLimit = cpuParams.threadLimit};
        }
        else if constexpr (target == Target::gpu)
        {
          [[maybe_unused]] const auto& gpuParams = getTargetConfig<Target::gpu>();

          return afft::gpu::Parameters
          {
#         if AFFT_GPU_BACKEND_IS_CUDA || AFFT_GPU_BACKEND_IS_HIP
            .device            = gpuParams.device,
            .externalWorkspace = gpuParams.externalWorkspace
#         endif
          };
        }
        else
        {
          unreachable();
        }
      }

      /// @brief Equality operator.
      [[nodiscard]] friend constexpr bool operator==(const Config&, const Config&) noexcept = default;

    private:
      /**
       * @brief Check common parameters.
       * @param commonParams The common parameters.
       * @return The common parameters.
       */
      [[nodiscard]] static const CommonParameters& checkCommonParameters(const CommonParameters& commonParams)
      {
        checkValid<isValidComplexFormat>(commonParams.complexFormat, "Invalid complexFormat");
        checkValid<isValidPlacement>(commonParams.placement, "Invalid placement");
        checkValid<isValidInitEffort>(commonParams.initEffort, "Invalid initEffort");
        checkValid<isValidNormalize>(commonParams.normalize, "Invalid normalize");
        checkValid<isValidWorkspacePolicy>(commonParams.workspacePolicy, "Invalid workspacePolicy");

        return commonParams;
      }

      CommonParameters mCommonParams{};  ///< Common parameters.
      DimensionsConfig mDimsConfig{};    ///< Dimensions configuration.
      TransformConfig  mTransformConfig; ///< Transform configuration.
      TargetConfig     mTargetConfig;    ///< Target configuration.
  };
} // namespace afft::detail

/**
 * @brief Hash specialization for Config.
 */
template<>
struct std::hash<afft::detail::Config>
{
  [[nodiscard]] constexpr std::size_t operator()(const afft::detail::Config&) const noexcept
  {
    std::size_t seed = 0;

    // TODO

    return seed;
  }
};

#endif /* AFFT_DETAIL_CONFIG_HPP */
