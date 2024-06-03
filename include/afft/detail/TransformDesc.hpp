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

#ifndef AFFT_DETAIL_TRANSFORM_DESC_HPP
#define AFFT_DETAIL_TRANSFORM_DESC_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "common.hpp"
#include "../type.hpp"
#include "../typeTraits.hpp"

namespace afft::detail
{
  /// @brief Description of a DFT transform.
  struct DftDesc
  {
    dft::Type type{}; ///< Type of the transform.
  };

  /// @brief Description of a DHT transform.
  struct DhtDesc
  {
    dht::Type type{}; ///< Type of the transform.
  };

  /// @brief Description of a DTT transform.
  struct DttDesc
  {
    MaxDimArray<dtt::Type> types{}; ///< Types of the transform.
  };

  /**
   * @brief Description of a transform.
   */
  class TransformDesc
  {
    public:
      /// @brief Default constructor is not allowed.
      TransformDesc() = delete;

      /**
       * @brief Constructor.
       * @tparam TransformParamsT Type of the transform parameters.
       * @param transformParams Transform parameters.
       */
      template<typename TransformParamsT>
      TransformDesc(const TransformParamsT& transformParams)
      : mDirection(validateAndReturn(transformParams.direction)),
        mPrecision(validateAndReturn(transformParams.precision)),
        mShapeRank(transformParams.shape.size()),
        mShape(makeShape(transformParams.shape)),
        mTransformRank(transformParams.axes.size()),
        mTransformAxes(makeTransformAxes(transformParams.axes, mShapeRank)),
        mNormalization(validateAndReturn(transformParams.normalization)),
        mPlacement(validateAndReturn(transformParams.placement)),
        mTransformVariant(makeTransformVariant(transformParams, mTransformRank))
      {}

      /// @brief Copy constructor.
      TransformDesc(const TransformDesc&) = default;

      /// @brief Move constructor.
      TransformDesc(TransformDesc&&) = default;

      /// @brief Destructor.
      ~TransformDesc() = default;

      /// @brief Copy assignment operator.
      TransformDesc& operator=(const TransformDesc&) = default;

      /// @brief Move assignment operator.
      TransformDesc& operator=(TransformDesc&&) = default;

      /**
       * @brief Get direction of the transform.
       * @return Direction of the transform.
       */
      [[nodiscard]] constexpr Direction getDirection() const noexcept
      {
        return mDirection;
      }

      /**
       * @brief Get precision triad of the transform.
       * @return Precision triad of the transform.
       */
      [[nodiscard]] constexpr const PrecisionTriad& getPrecision() const noexcept
      {
        return mPrecision;
      }

      /**
       * @brief Get the rank of the shape.
       * @return Rank of the shape.
       */
      [[nodiscard]] constexpr std::size_t getShapeRank() const noexcept
      {
        return mShapeRank;
      }

      /**
       * @brief Get the shape of the transform.
       * @return Shape of the transform.
       */
      [[nodiscard]] constexpr View<std::size_t> getShape() const noexcept
      {
        return View<std::size_t>{mShape.data(), mShapeRank};
      }

      /**
       * @brief Convert the shape to a different integral type.
       * @tparam I Integral type.
       * @return Shape of the transform as a different integral type. Only first getShapeRank() elements are valid.
       */
      template<typename I>
      [[nodiscard]] constexpr MaxDimArray<I> getShapeAs() const
      {
        static_assert(std::is_integral_v<I>, "Integral type required");

        return shape.cast<I>();
      }

      /**
       * @brief Get the rank of the transform.
       * @return Rank of the transform.
       */
      [[nodiscard]] constexpr std::size_t getTransformRank() const noexcept
      {
        return mTransformRank;
      }

      /**
       * @brief Get the axes of the transform.
       * @return Axes of the transform.
       */
      [[nodiscard]] constexpr View<std::size_t> getTransformAxes() const noexcept
      {
        return View<std::size_t>{mTransformAxes.data(), mTransformRank};
      }

      /**
       * @brief Get the dimensions of the transform.
       * @tparam I Integral type.
       * @return Dimensions of the transform.
       */
      template<typename I>
      [[nodiscard]] constexpr MaxDimArray<I> getTransformDimsAs() const
      {
        static_assert(std::is_integral_v<I>, "Integral type required");

        MaxDimArray<I> dims{};

        for (std::size_t i{}; i < getTransformRank(); ++i)
        {
          dims[i] = safeIntCast<I>(mShape[mTransformAxes[i]]);
        }

        return dims;
      }

      /**
       * @brief Get the axes of the transform as a different integral type.
       * @tparam I Integral type.
       * @return Axes of the transform as a different integral type. Only first getTransformRank() elements are valid.
       */
      template<typename I>
      [[nodiscard]] constexpr MaxDimArray<I> getTransformHowManyDimsAs() const
      {
        static_assert(std::is_integral_v<I>, "Integral type required");

        MaxDimArray<I> dims{};

        const auto axes = getAxes();

        for (std::size_t i{}, pos{}; i < getShapeRank(); ++i)
        {
          std::size_t j{};

          for (; j < getTransformRank(); ++j)
          {
            if (axes[j] == i)
            {
              break;
            }
          }

          if (j == getTransformRank())
          {
            dims[pos++] = safeIntCast<I>(mShape[i]);
          }
        }

        return dims;
      }

      /**
       * @brief Get the normalization of the transform.
       * @return Normalization of the transform.
       */
      [[nodiscard]] constexpr Normalization getNormalization() const noexcept
      {
        return mNormalization;
      }

      /**
       * @brief Get the placement of the transform.
       * @return Placement of the transform.
       */
      [[nodiscard]] constexpr Placement getPlacement() const noexcept
      {
        return mPlacement;
      }

      /**
       * @brief Get the transform type.
       * @return Transform type.
       */
      [[nodiscard]] constexpr Transform getTransform() const
      {
        if (std::holds_alternative<DftDesc>(mTransformVariant))
        {
          return Transform::dft;
        }
        else if (std::holds_alternative<DhtDesc>(mTransformVariant))
        {
          return Transform::dht;
        }
        else if (std::holds_alternative<DttDesc>(mTransformVariant))
        {
          return Transform::dtt;
        }
        else
        {
          throw std::runtime_error("Invalid transform variant state");
        }
      }

      /**
       * @brief Get the transform description.
       * @tparam transform Transform type.
       * @return Transform description.
       */
      template<Transform transform>
      [[nodiscard]] constexpr auto getTransformDesc()
      {
        static_assert(isValid(transform), "Invalid transform type");

        if constexpr (transform == Transform::dft)
        {
          return std::get<DftDesc>(mTransformVariant);
        }
        else if constexpr (transform == Transform::dht)
        {
          return std::get<DhtDesc>(mTransformVariant);
        }
        else if constexpr (transform == Transform::dtt)
        {
          return std::get<DttDesc>(mTransformVariant);
        }
        else
        {
          cxx::unreachable();
        }
      }

      /**
       * @brief Get the normalization factor.
       * @tparam prec Precision type.
       * @return Normalization factor.
       */
      template<Precision prec>
      [[nodiscard]] constexpr Real<prec> getNormalizationFactor() const
      {
        static_assert(isValid(prec), "Invalid precision type");

        std::size_t logicalSize{1};

        switch (getTransform())
        {
          case Transform::dft:
          case Transform::dht:
            for (const auto& axis : getAxes())
            {
              logicalSize *= getShape()[axis];
            }
            break;
          case Transform::dtt:
          {
            const auto& dttDesc = getTransformDesc<Transform::dtt>();

            for (std::size_t i{}; i < getTransformRank(); ++i)
            {
              switch (dttDesc.axisTypes[i])
              {
                case dtt::Type::dct1:
                  logicalSize *= 2 * (getShape()[getAxes()[i]] - 1);
                  break;
                case dtt::Type::dst1:
                  logicalSize *= 2 * (getShape()[getAxes()[i]] + 1);
                  break;
                case dtt::Type::dct2:
                case dtt::Type::dct3:
                case dtt::Type::dct4:
                case dtt::Type::dst2:
                case dtt::Type::dst3:
                case dtt::Type::dst4:
                  logicalSize *= 2 * getShape()[getAxes()[i]];
                  break;
                default:
                  cxx::unreachable();
              }
            }
            break;
          }
          default:
            cxx::unreachable();
        }

        switch (getNormalization())
        {
          case Normalization::none:
            return Real<prec>{1};
          case Normalization::orthogonal:
            return Real<prec>{1} / std::sqrt(static_cast<Real<prec>>(logicalSize));
          case Normalization::unitary:
            return Real<prec>{1} / static_cast<Real<prec>>(logicalSize);
          default:
            cxx::unreachable();
        }
      }

      /**
       * @brief Get the transform parameters.
       * @tparam transform Transform type.
       * @return Transform parameters.
       */
      template<Transform transform>
      [[nodiscard]] constexpr TransformParameters<transform> getTransformParameters() const
      {
        static_assert(isValid(transform), "Invalid transform type");

        TransformParameters<transform> transformParams{};

        transformParams.direction     = getDirection();
        transformParams.precision     = getPrecision();
        transformParams.shape         = getShape();
        transformParams.axes          = getTransformAxes();
        transformParams.normalization = getNormalization();

        if constexpr (transform == Transform::dft)
        {
          transformParams.type = getTransformDesc<Transform::dft>().type;
        }
        else if constexpr (transform == Transform::dht)
        {
          transformParams.type = getTransformDesc<Transform::dht>().type;
        }
        else if constexpr (transform == Transform::dtt)
        {
          transformParams.types = View<dtt::Type>{getTransformDesc<Transform::dtt>().types.data(), getTransformRank()};
        }
        
        return transformParams;
      }
    private:
      /// @brief Transform variant type.
      using TransformVariant = std::variant<DftDesc, DhtDesc, DttDesc>;

      /**
       * @brief Make the shape of the transform.
       * @param shapeView Shape view.
       * @return Shape of the transform.
       */
      [[nodiscard]] constexpr static MaxDimArray<std::size_t> makeShape(View<std::size_t> shapeView)
      {
        MaxDimArray<std::size_t> shape{};

        if (shapeView.size() > maxDimCount)
        {
          throw std::invalid_argument("Too many shape dimensions");
        }
        else if (shapeView.size() == 0)
        {
          throw std::invalid_argument("Empty shape");
        }

        for (std::size_t i{}; i < shapeView.size(); ++i)
        {
          if (shapeView[i] == 0)
          {
            throw std::invalid_argument("Invalid shape dimension size");
          }

          shape[i] = shapeView[i];
        }

        return shape;
      }

      /**
       * @brief Make the transform axes.
       * @param axesView Axes view.
       * @param shapeRank Rank of the shape.
       * @return Transform axes.
       */
      [[nodiscard]] static MaxDimArray<std::size_t>
      makeTransformAxes(View<std::size_t> axesView, std::size_t shapeRank)
      {
        MaxDimArray<std::size_t> axes{};

        if (axesView.empty())
        {
          std::iota(axes.begin(), axes.begin() + shapeRank, 0);
        }
        else if (axesView.size() <= shapeRank)
        {
          std::bitset<maxDimCount> seenAxes{};
          
          for (const auto& axis : axes)
          {
            if (axis >= shapeRank)
            {
              throw std::invalid_argument("Transform axis out of bounds");
            }
            else if (seenAxes.test(axis))
            {
              throw std::invalid_argument("Transform axes must be unique");
            }

            seenAxes.set(axis);
          }
        }
        else
        {
          throw std::invalid_argument("Too many transform axes");
        }

        return axes;
      }

      /**
       * @brief Make the transform variant.
       * @tparam sRank Rank of the shape.
       * @tparam tRank Rank of the transform.
       * @param dftParams DFT parameters.
       * @param transformRank Rank of the transform.
       * @return Transform variant.
       */
      template<std::size_t sRank, std::size_t tRank>
      [[nodiscard]] static TransformVariant
      makeTransformVariant(const dft::Parameters<sRank, tRank>& dftParams, std::size_t)
      {
        return DftDesc{validateAndReturn(dftParams.type)};
      }

      /**
       * @brief Make the transform variant.
       * @tparam sRank Rank of the shape.
       * @tparam tRank Rank of the transform.
       * @param dhtParams DHT parameters.
       * @param transformRank Rank of the transform.
       * @return Transform variant.
       */
      template<std::size_t sRank, std::size_t tRank>
      [[nodiscard]] static TransformVariant
      makeTransformVariant(const dht::Parameters<sRank, tRank>& dhtParams, std::size_t)
      {
        return DhtDesc{validateAndReturn(dhtParams.type)};
      }

      /**
       * @brief Make the transform variant.
       * @tparam sRank Rank of the shape.
       * @tparam tRank Rank of the transform.
       * @tparam ttRank Rank of the transform type.
       * @param dttParams DTT parameters.
       * @param transformRank Rank of the transform.
       * @return Transform variant.
       */
      template<std::size_t sRank, std::size_t tRank, std::size_t ttRank>
      [[nodiscard]] static TransformVariant
      makeTransformVariant(const dtt::Parameters<sRank, tRank, ttRank>& dttParams, std::size_t transformRank)
      {
        if ((transformRank != 1) && (dttParams.types.size() != transformRank))
        {
          throw std::invalid_argument("Invalid number of dtt types, must be 1 or equal to the number of axes");
        }

        DttDesc dttDesc{};

        for (std::size_t i{}; i < transformRank; ++i)
        {
          dttDesc.types[i] = validateAndReturn(dttParams.types[(transformRank == 1) ? 0 : i]);
        }

        return dttDesc;
      }

      Direction                mDirection{};      ///< Direction of the transform.
      PrecisionTriad           mPrecision{};      ///< Precision triad of the transform.
      std::size_t              mShapeRank{};      ///< Rank of the shape.
      MaxDimArray<std::size_t> mShape{};          ///< Shape of the transform.
      std::size_t              mTransformRank{};  ///< Rank of the transform.
      MaxDimArray<std::size_t> mTransformAxes{};  ///< Axes of the transform.
      Normalization            mNormalization{};  ///< Normalization of the transform.
      Placement                mPlacement{};      ///< Placement of the transform.
      TransformVariant         mTransformVariant; ///< Transform variant.
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_TRANSFORM_DESC_HPP */
