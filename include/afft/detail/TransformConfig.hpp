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

#ifndef AFFT_DETAIL_TRANSFORM_CONFIG_HPP
#define AFFT_DETAIL_TRANSFORM_CONFIG_HPP

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <variant>
#include <span>

#include "common.hpp"
#include "error.hpp"
#include "utils.hpp"

namespace afft::detail
{
  /**
   * @brief Configuration for a Discrete Fourier Transform (DFT) transform.
   */
  struct DftConfig
  {
    dft::Type type{}; ///< DFT transform type.

    /// @brief Default equality operator.
    friend constexpr bool operator==(const DftConfig&, const DftConfig&) = default;
  };

  /**
   * @brief Configuration for a Discrete Trigonometric Transform (DTT) transform.
   */
  struct DttConfig
  {
    MaxDimArray<dtt::Type> axisTypes{}; ///< DTT transform types for each axis.

    friend constexpr bool operator==(const DttConfig&, const DttConfig&) = default;
  };

  /**
   * @brief Configuration for a transform.
   */
  class TransformConfig
  {
    public:
      /// @brief Default constructor not allowed.
      TransformConfig() = delete;

      /**
       * @brief Create a transform configuration.
       * @param axes Transform axes.
       * @param variant Transform variant.
       */
      constexpr TransformConfig(const dft::Parameters& dftParams)
      : mDirection{checkDirection(dftParams.direction)},
        mPrec{checkPrecision(dftParams.precision)},
        mRank{(dftParams.axes.empty()) ? dftParams.dimensions.shape.size() : dftParams.axes.size()},
        mAxes{checkAxes(dftParams.axes, dftParams.dimensions.shape.size())},
        mVariant{makeTransformVariant(dftParams)}
      {}

      /// @brief Copy constructor.
      constexpr TransformConfig(const TransformConfig&) noexcept = default;

      /// @brief Move constructor.
      constexpr TransformConfig(TransformConfig&&) noexcept = default;

      /// @brief Destructor.
      ~TransformConfig() noexcept = default;

      /// @brief Copy assignment operator.
      constexpr TransformConfig& operator=(const TransformConfig&) noexcept = default;
      
      /// @brief Move assignment operator.
      constexpr TransformConfig& operator=(TransformConfig&&) noexcept = default;

      /**
       * @brief Get the transform direction.
       * @return Transform direction.
       */
      [[nodiscard]] constexpr Direction getDirection() const noexcept
      {
        return mDirection;
      }

      /**
       * @brief Get the transform precision.
       * @return Transform precision.
       */
      [[nodiscard]] constexpr const PrecisionTriad& getPrecision() const noexcept
      {
        return mPrec;
      }

      /**
       * @brief Get the transform rank.
       * @return Transform rank.
       */
      [[nodiscard]] constexpr std::size_t getRank() const noexcept
      {
        return mRank;
      }

      /**
       * @brief Get the transform axes.
       * @return Transform axes.
       */
      [[nodiscard]] constexpr std::span<const std::size_t> getAxes() const noexcept
      {
        return {mAxes.begin(), mRank};
      }

      /**
       * @brief Get transform type.
       * @return Transform type.
       */
      [[nodiscard]] constexpr Transform getType() const noexcept
      {
        return static_cast<Transform>(mVariant.index());
      }

      /**
       * @brief Get transform configuration.
       * @tparam transform Transform type.
       * @return Transform configuration.
       */
      template<Transform transform>
      [[nodiscard]] constexpr const auto& getConfig() const noexcept
      {
        if constexpr (transform == Transform::dft)
        {
          return std::get<DftConfig>(mVariant);
        }
        else if constexpr (transform == Transform::dtt)
        {
          return std::get<DttConfig>(mVariant);
        }
      }

      /**
       * @brief Get the logical size of the transform.
       * @param dims Dimensions.
       * @return Logical size.
       */
      [[nodiscard]] constexpr std::size_t getTransformLogicalSize(std::span<const std::size_t> dims) const
      {
        std::size_t logicalSize{1};

        for (std::size_t i{}; i < mRank; ++i)
        {
          const auto n = dims[mAxes[i]];

          switch (getType())
          {
          case Transform::dft:
            logicalSize *= n;
            break;
          case Transform::dtt:
          {
            const auto& dttParams = getConfig<Transform::dtt>();

            switch (dttParams.axisTypes[i])
            {
              using enum dtt::Type;
            case dct1:                       logicalSize *= 2 * (n - 1); break;
            case dst1:                       logicalSize *= 2 * (n + 1); break;
            case dct2: case dct3: case dct4:
            case dst2: case dst3: case dst4: logicalSize *= 2 * n;       break;
            default:
              unreachable();
            }
            break;
          }
          default:
            unreachable();
          }
        }

        return logicalSize;
      }

      /**
       * @brief Corrects the dimensions configuration.
       * @param dimsConfig Dimensions configuration.
       * @param commonParams Common parameters.
       */
      void correctDimensionsConfig(DimensionsConfig& dimsConfig, const CommonParameters& commonParams) const
      {
        auto generateStrides = [&](std::span<std::size_t> strides, std::invocable<std::size_t, std::size_t> auto fn)
        {
          for (std::size_t i{}; i < dimsConfig.getRank(); ++i)
          {
            if (i == 0)
            {
              strides.back() = 1;
            }
            else
            {
              std::size_t axis = dimsConfig.getRank() - i - 1;

              strides[axis] = fn(axis + 1, strides[axis + 1]);
            }
          }
        };

        auto defaultStrideGenerator = [dimsConfig](std::size_t axis, std::size_t prevStride) -> std::size_t
        {
          return dimsConfig.getShape()[axis] * prevStride;
        };

        auto dftHermitComplexStrideGenerator = [dimsConfig, redAxis = mAxes.back()]
                                               (std::size_t axis, std::size_t prevStride) -> std::size_t
        {
          const auto size = dimsConfig.getShape()[axis];

          return ((axis == redAxis) ? size / 2 + 1 : size) * prevStride;
        };

        auto dftRealStrideGenerator = [dimsConfig, redAxis = mAxes.back(), placement = commonParams.placement]
                                      (std::size_t axis, std::size_t prevStride) -> std::size_t
        {
          const auto size = dimsConfig.getShape()[axis];

          return ((placement == Placement::inPlace && axis == redAxis) ? 2 * (size / 2 + 1) : size) * prevStride;
        };

        if (!dimsConfig.hasSrcStride())
        {
          switch (getType())
          {
          case Transform::dft:
          {
            switch (getConfig<Transform::dft>().type)
            {
            case dft::Type::complexToReal:
              generateStrides(dimsConfig.getSrcStrides(), dftHermitComplexStrideGenerator);
              break;
            case dft::Type::realToComplex:
              generateStrides(dimsConfig.getSrcStrides(), dftRealStrideGenerator);
              break;
            default:
              generateStrides(dimsConfig.getSrcStrides(), defaultStrideGenerator);
              break;
            }
            break;
          }
          default:
            generateStrides(dimsConfig.getSrcStrides(), defaultStrideGenerator);
            break;
          }
        }

        if (!dimsConfig.hasDstStride())
        {
          switch (getType())
          {
          case Transform::dft:
          {
            switch (getConfig<Transform::dft>().type)
            {
            case dft::Type::realToComplex:
              generateStrides(dimsConfig.getDstStrides(), dftHermitComplexStrideGenerator);
              break;
            case dft::Type::complexToReal:
              generateStrides(dimsConfig.getDstStrides(), dftRealStrideGenerator);
              break;
            default:
              generateStrides(dimsConfig.getDstStrides(), defaultStrideGenerator);
              break;
            }
            break;
          }
          default:
            generateStrides(dimsConfig.getDstStrides(), defaultStrideGenerator);
            break;
          }
        }
      }

      /**
       * @brief Check the execution types.
       * @param srcPrec Source precision.
       * @param srcCmpl Source complexity.
       * @param dstPrec Destination precision.
       * @param dstCmpl Destination complexity.
       */
      constexpr void checkExecTypes(Precision srcPrec, Complexity srcCmpl, Precision dstPrec, Complexity dstCmpl) const
      {
        if (srcPrec != mPrec.source)
        {
          throw makeException<std::invalid_argument>("Invalid source precision for transform");
        }
        
        if (dstPrec != mPrec.destination)
        {
          throw makeException<std::invalid_argument>("Invalid destination precision for transform");
        }

        Complexity refSrcCmpl{};
        Complexity refDstCmpl{};

        switch (getType())
        {
        case Transform::dft:
        {
          const auto& dftParams = getConfig<Transform::dft>();

          switch (dftParams.type)
          {
          case dft::Type::complexToComplex:
            refSrcCmpl = Complexity::complex;
            refDstCmpl = Complexity::complex;
            break;
          case dft::Type::realToComplex:
            refSrcCmpl = Complexity::real;
            refDstCmpl = Complexity::complex;
            break;
          case dft::Type::complexToReal:
            refSrcCmpl = Complexity::complex;
            refDstCmpl = Complexity::real;
            break;
          default:
            unreachable();
          }
          break;
        }
        case Transform::dtt:
          refSrcCmpl = Complexity::real;
          refDstCmpl = Complexity::real;
          break;
        default:
          throw makeException<std::runtime_error>("Invalid transform type");
        }

        if (srcCmpl != refSrcCmpl)
        {
          throw makeException<std::invalid_argument>("Invalid source complexity");
        }

        if (dstCmpl != refDstCmpl)
        {
          throw makeException<std::invalid_argument>("Invalid destination complexity");
        }
      }
      
      /**
       * @brief Equality operator.
       * @param lhs Left-hand side.
       * @param rhs Right-hand side.
       * @return True if equal, false otherwise.
       */
      [[nodiscard]] friend constexpr bool operator==(const TransformConfig& lhs, const TransformConfig& rhs)
      {
        if (lhs.mRank != rhs.mRank)
        {
          return false;
        }

        for (std::size_t i{}; i < lhs.mRank; ++i)
        {
          if (lhs.mAxes[i] != rhs.mAxes[i])
          {
            return false;
          }
        }

        return lhs.mVariant == rhs.mVariant;
      }

      /**
       * @brief Inequality operator.
       * @param lhs Left-hand side.
       * @param rhs Right-hand side.
       * @return True if not equal, false otherwise.
       */
      [[nodiscard]] friend constexpr bool operator!=(const TransformConfig& lhs, const TransformConfig& rhs)
      {
        return !(lhs == rhs);
      }
    protected:
    private:
      using ConfigVariant = std::variant<DftConfig, DttConfig>;

      /**
       * @brief Check transform direction validity.
       * @param dir Transform direction.
       * @return Direction.
       */
      [[nodiscard]] static constexpr Direction checkDirection(Direction dir)
      {
        checkValid<isValidDirection>(dir, "Invalid transform direction");

        return dir;
      }

      /**
       * @brief Check transform precision validity.
       * @param prec Transform precision.
       * @return Precision.
       */
      [[nodiscard]] static constexpr const PrecisionTriad& checkPrecision(const PrecisionTriad& prec)
      {
        checkValid<isValidPrecision>(prec.execution, "Invalid execution precision");
        checkValid<isValidPrecision>(prec.source, "Invalid source precision");
        checkValid<isValidPrecision>(prec.destination, "Invalid destination precision");

        if (!hasPrecision(prec.execution) || !hasPrecision(prec.source) || !hasPrecision(prec.destination))
        {
          throw makeException<std::invalid_argument>("Invalid transform precision");
        }

        return prec;
      }

      /**
       * @brief Check transform axes.
       * @param axes Transform axes.
       * @param shapeRank Shape rank.
       */
      [[nodiscard]] static constexpr MaxDimArray<std::size_t>
      checkAxes(std::span<const std::size_t> axes, std::size_t shapeRank)
      {
        if (axes.empty())
        {
          MaxDimArray<std::size_t> axesArray{};

          for (std::size_t i{}; i < shapeRank; ++i)
          {
            axesArray[i] = i;
          }

          return axesArray;
        }
        else if (axes.size() > shapeRank)
        {
          throw makeException<std::invalid_argument>("Transform axes rank exceeds shape rank");
        }
        else if (axes.size() > maxDimCount)
        {
          throw makeException<std::invalid_argument>("Transform axes rank exceeds maximum rank");
        }

        std::bitset<maxDimCount> seenAxes{};
        
        for (const auto& axis : axes)
        {
          if (axis >= shapeRank)
          {
            throw makeException<std::invalid_argument>("Transform axis out of bounds");
          }
          else if (seenAxes.test(axis))
          {
            throw makeException<std::invalid_argument>("Transform axes must be unique");
          }

          seenAxes.set(axis);
        }

        MaxDimArray<std::size_t> axesArray{};

        std::copy(axes.begin(), axes.end(), axesArray.begin());

        return axesArray;
      }
      
      /**
       * @brief Create a transform variant.
       * @param dftParams DFT parameters.
       * @return Transform variant.
       */
      [[nodiscard]] static constexpr DftConfig makeTransformVariant(const dft::Parameters& dftParams)
      {
        checkValid<isValidDftType>(dftParams.type, "Invalid dft transform type");
        
        return DftConfig{dftParams.type};
      }

      /**
       * @brief Create a transform variant.
       * @param dttParams DTT parameters.
       * @return Transform variant.
       */
      [[nodiscard]] static constexpr DttConfig makeTransformVariant(const dtt::Parameters& dttParams)
      {
        checkValid<isValidDttType>(dttParams.types, "Invalid dtt transform types");

        DttConfig dttConfig{};

        if (dttParams.types.size() == 1)
        {
          std::fill_n(dttConfig.axisTypes.begin(), dttParams.axes.size(), dttParams.types[0]);
        }
        else if (dttParams.types.size() == dttParams.axes.size())
        {
          std::copy(dttParams.types.begin(), dttParams.types.end(), dttConfig.axisTypes.begin());
        }
        else
        {
          throw makeException<std::invalid_argument>("Invalid dtt transform types");
        }

        return dttConfig;
      }

      Direction                mDirection{}; ///< Transform direction.
      PrecisionTriad           mPrec{};      ///< Transform precision.
      std::size_t              mRank{};      ///< Transform rank.
      MaxDimArray<std::size_t> mAxes{};      ///< Transform axes.
      ConfigVariant            mVariant{};   ///< Transform variant.
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_TRANSFORM_CONFIG_HPP */
