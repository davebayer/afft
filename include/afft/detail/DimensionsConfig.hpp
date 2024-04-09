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

#ifndef AFFT_DETAIL_DIMENSIONS_CONFIG_HPP
#define AFFT_DETAIL_DIMENSIONS_CONFIG_HPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <span>

#include "common.hpp"

namespace afft::detail
{
  /**
   * @class DimensionsConfig
   * @brief Configuration for dimensions.
   */
  class DimensionsConfig
  {
    public:
      /// @brief Default constructor.
      constexpr DimensionsConfig() = default;
      /**
       * @brief Constructor.
       * @param dims Dimensions.
       */
      constexpr DimensionsConfig(const Dimensions& dims)
      : mRank{dims.shape.size()}
      {
        auto assertNotZero = [](const std::size_t elem) constexpr -> std::size_t
        {
          if (elem == std::size_t{})
          {
            throw makeException<std::runtime_error>("Invalid dimension size");
          }

          return elem;
        };

        // Ensure rank is within bounds
        if (mRank > maxDimCount)
        {
          throw makeException<std::runtime_error>("Too many dimensions");
        }

        // Copy shape and ensure all dimensions are non-zero
        std::transform(dims.shape.begin(), dims.shape.end(), mShape.begin(), assertNotZero);

        // Copy source strides if provided, otherwise leave initialized to 0
        if (dims.srcStride.size() == mRank)
        {
          std::copy(dims.srcStride.begin(), dims.srcStride.end(), mSrcStride.begin());
        }
        else if (!dims.srcStride.empty())
        {
          throw makeException<std::runtime_error>("Invalid source stride size");
        }

        // Copy destination strides if provided, otherwise leave initialized to 0
        if (dims.dstStride.size() == mRank)
        {
          std::copy(dims.dstStride.begin(), dims.dstStride.end(), mDstStride.begin());
        }
        else if (!dims.dstStride.empty())
        {
          throw makeException<std::runtime_error>("Invalid destination stride size");
        }
      }
      /// @brief Copy constructor.
      constexpr DimensionsConfig(const DimensionsConfig&) = default;
      /// @brief Move constructor.
      constexpr DimensionsConfig(DimensionsConfig&&) = default;
      /// @brief Destructor.
      ~DimensionsConfig() = default;

      /// @brief Copy assignment operator.
      constexpr DimensionsConfig& operator=(const DimensionsConfig&) = default;
      /// @brief Move assignment operator.
      constexpr DimensionsConfig& operator=(DimensionsConfig&&) = default;

      /**
       * @brief Get shape rank.
       * @return Rank.
       */
      [[nodiscard]] constexpr std::size_t getRank() const noexcept
      {
        return mRank;
      }

      /**
       * @brief Get shape.
       * @return Shape.
       */
      [[nodiscard]] constexpr std::span<std::size_t> getShape() noexcept
      {
        return {mShape.data(), mRank};
      }

      /**
       * @brief Get shape.
       * @return Shape.
       */
      [[nodiscard]] constexpr std::span<const std::size_t> getShape() const noexcept
      {
        return {mShape.data(), mRank};
      }

      /**
       * @brief Get source stride.
       * @return Source stride.
       */
      [[nodiscard]] constexpr std::span<std::size_t> getSrcStride() noexcept
      {
        return {mSrcStride.data(), mRank};
      }

      /**
       * @brief Get source stride.
       * @return Source stride.
       */
      [[nodiscard]] constexpr std::span<const std::size_t> getSrcStride() const noexcept
      {
        return {mSrcStride.data(), mRank};
      }

      /**
       * @brief Get destination stride.
       * @return Destination stride.
       */
      [[nodiscard]] constexpr std::span<std::size_t> getDstStride() noexcept
      {
        return {mDstStride.data(), mRank};
      }

      /**
       * @brief Get destination stride.
       * @return Destination stride.
       */
      [[nodiscard]] constexpr std::span<const std::size_t> getDstStride() const noexcept
      {
        return {mDstStride.data(), mRank};
      }

      /**
       * @brief Equality operator.
       * @param lhs Left-hand side.
       * @param rhs Right-hand side.
       * @return True if equal, false otherwise.
       */
      [[nodiscard]] friend constexpr bool operator==(const DimensionsConfig& lhs, const DimensionsConfig& rhs) noexcept
      {
        if (lhs.mRank != rhs.mRank)
        {
          return false;
        }

        for (std::size_t i = 0; i < lhs.mRank; ++i)
        {
          if (lhs.mShape[i] != rhs.mShape[i])
          {
            return false;
          }

          if (lhs.mSrcStride[i] != rhs.mSrcStride[i])
          {
            return false;
          }

          if (lhs.mDstStride[i] != rhs.mDstStride[i])
          {
            return false;
          }
        }

        return true;
      }

      /**
       * @brief Inequality operator.
       * @param lhs Left-hand side.
       * @param rhs Right-hand side.
       * @return True if not equal, false otherwise.
       */
      [[nodiscard]] friend constexpr bool operator!=(const DimensionsConfig& lhs, const DimensionsConfig& rhs) noexcept
      {
        return !(lhs == rhs);
      }
    private:
      std::size_t              mRank{};      ///< Rank.
      MaxDimArray<std::size_t> mShape{};     ///< Shape.
      MaxDimArray<std::size_t> mSrcStride{}; ///< Source stride.
      MaxDimArray<std::size_t> mDstStride{}; ///< Destination stride.
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_DIMENSIONS_CONFIG_HPP */
