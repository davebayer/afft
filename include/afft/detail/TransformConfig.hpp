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
    dft::Format srcFormat{}; ///< Source data format.
    dft::Format dstFormat{}; ///< Destination data format.

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
      /**
       * @brief Create a transform configuration from DFT parameters.
       * @param dftParams DFT parameters.
       * @return Transform configuration.
       */
      [[nodiscard]] static constexpr TransformConfig make(const dft::Parameters& dftParams)
      {
        checkAxes(dftParams.axes, dftParams.dimensions.shape.size());

        bool formatsOk = true;

        switch (dftParams.srcFormat)
        {
          using enum dft::Format;
        case real:
          switch (dftParams.dstFormat)
          {
          case complexInterleaved:
          case complexPlanar:      break;
          default:                 formatsOk = false; break;
          }
          break;
        case complexInterleaved:
        case complexPlanar:
          break;
        default:
          formatsOk = false;
          break;
        }

        if (!formatsOk)
        {
          throw makeException<std::invalid_argument>("Invalid dft transform formats combination");
        }

        return TransformConfig{dftParams.axes, DftConfig{dftParams.srcFormat, dftParams.dstFormat}};
      }

      /**
       * @brief Create a transform configuration from DTT parameters.
       * @param dttParams DTT parameters.
       * @return Transform configuration.
       */
      [[nodiscard]] static constexpr TransformConfig make(const dtt::Parameters& dttParams)
      {
        auto checkDttType = [](dtt::Type type) constexpr -> void
        {
          switch (type)
          {
            using enum dtt::Type;
          case dct1: case dct2: case dct3: case dct4:
          case dst1: case dst2: case dst3: case dst4:
            break;
          default:
            throw makeException<std::invalid_argument>("Invalid dtt transform type");
          }
        };

        DttConfig dttConfig{};

        checkAxes(dttParams.axes, dttParams.dimensions.shape.size());

        if (dttParams.types.size() == 1)
        {
          checkDttType(dttParams.types[0]);
          std::fill_n(dttConfig.axisTypes.begin(), dttParams.axes.size(), dttParams.types[0]);
        }
        else if (dttParams.types.size() == dttParams.axes.size())
        {
          std::for_each(dttParams.types.begin(), dttParams.types.end(), checkDttType);
          std::copy(dttParams.types.begin(), dttParams.types.end(), dttConfig.axisTypes.begin());
        }
        else
        {
          throw makeException<std::invalid_argument>("Invalid dtt transform types");
        }

        return TransformConfig{dttParams.axes, std::move(dttConfig)};
      }

      /// @brief Default constructor not allowed.
      TransformConfig() = delete;
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
      [[nodiscard]] constexpr TransformType getType() const noexcept
      {
        return static_cast<TransformType>(mVariant.index());
      }

      /**
       * @brief Get transform configuration.
       * @tparam transformType Transform type.
       * @return Transform configuration.
       */
      template<TransformType transformType>
      [[nodiscard]] constexpr const auto& getConfig() const noexcept
      {
        if constexpr (transformType == TransformType::dft)
        {
          return std::get<DftConfig>(mVariant);
        }
        else if constexpr (transformType == TransformType::dtt)
        {
          return std::get<DttConfig>(mVariant);
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
      using ConfigVariant = std::variant<DftConfig,      // same index as TransformType::dft
                                                  DttConfig>;     // same index as TransformType::dtt

      static_assert(variant_alternative_index<ConfigVariant, DftConfig>()
                      == to_underlying(TransformType::dft));
      static_assert(variant_alternative_index<ConfigVariant, DttConfig>()
                      == to_underlying(TransformType::dtt));

      /**
       * @brief Check transform axes.
       * @param axes Transform axes.
       * @param shapeRank Shape rank.
       */
      static void checkAxes(std::span<const std::size_t> axes, std::size_t shapeRank)
      {
        if (axes.empty())
        {
          throw makeException<std::invalid_argument>("Transform axes cannot be empty");
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
      }

      /**
       * @brief Create a transform configuration.
       * @param axes Transform axes.
       * @param variant Transform variant.
       */
      constexpr TransformConfig(std::span<const std::size_t> axes, auto&& variant) noexcept
      : mRank{axes.size()}, mVariant{std::forward<decltype(variant)>(variant)}
      {
        std::copy(axes.begin(), axes.end(), mAxes.begin());
      }

      std::size_t              mRank{};    ///< Transform rank.
      MaxDimArray<std::size_t> mAxes{};    ///< Transform axes.
      ConfigVariant            mVariant{}; ///< Transform variant.
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_TRANSFORM_CONFIG_HPP */
