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

#ifndef AFFT_TRANSFORM_HPP
#define AFFT_TRANSFORM_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "common.hpp"
#include "detail/transform.hpp"

namespace afft
{
  /// @brief Named constant representing all axes (is empty view)
  template<std::size_t transformExt = dynamicRank>
  inline constexpr View<std::size_t, transformExt> allAxes{};

  /// @brief Namespace for discrete Fourier transform
  namespace dft
  {
    /// @brief DFT transform type
    enum class Type
    {
      complexToComplex,       ///< complex-to-complex transform
      realToComplex,          ///< real-to-complex transform
      complexToReal,          ///< complex-to-real transform

      c2c = complexToComplex, ///< alias for complex-to-complex transform
      r2c = realToComplex,    ///< alias for real-to-complex transform
      c2r = complexToReal,    ///< alias for complex-to-real transform
    };

    /**
     * @brief DFT Parameters
     * @tparam shapeExt Extent of the shape, dynamic by default
     * @tparam transformExt Extent of the transform, dynamic by default
     */
    template<std::size_t shapeExt = dynamicExtent, std::size_t transformExt = dynamicExtent>
    struct Parameters : detail::TransformParametersBase<shapeExt, transformExt>
    {
      static_assert((shapeExt == dynamicExtent) || (shapeExt > 0), "shape extent must be greater than 0");
      static_assert((transformExt == dynamicExtent) || (transformExt > 0), "transform extent must be greater than 0");
      static_assert((shapeExt == dynamicExtent) || (transformExt == dynamicExtent) || (transformExt <= shapeExt),
                    "transform rank must be less than or equal to shape rank");

      Direction                       direction{};                        ///< direction of the transform
      PrecisionTriad                  precision{};                        ///< precision triad
      View<std::size_t, shapeExt>     shape{};                            ///< shape of the transform
      View<std::size_t, transformExt> axes{allAxes<transformExt>};        ///< axes of the transform
      Normalization                   normalization{Normalization::none}; ///< normalization
      Placement                       placement{Placement::outOfPlace};   ///< placement of the transform
      Type                            type{Type::complexToComplex};       ///< type of the transform
    };
  } // namespace dft

  /// @brief Namespace for discrete Hartley transform
  namespace dht
  {
    /// @brief DHT transform type
    enum class Type
    {
      separable, ///< separable DHT, computes the DHT along each axis independently
    };

    /**
     * @brief DHT Parameters
     * @tparam shapeExt Extent of the shape, dynamic by default
     * @tparam transformExt Extent of the transform, dynamic by default
     */
    template<std::size_t shapeExt = dynamicExtent, std::size_t transformExt = dynamicExtent>
    struct Parameters : detail::TransformParametersBase<shapeExt, transformExt>
    {
      static_assert((shapeExt == dynamicExtent) || (shapeExt > 0), "shape rank must be greater than 0");
      static_assert((transformExt == dynamicExtent) || (transformExt > 0), "transform rank must be greater than 0");
      static_assert((shapeExt == dynamicExtent) || (transformExt == dynamicExtent) || (transformExt <= shapeExt),
                    "transform rank must be less than or equal to shape rank");

      Direction                       direction{};                        ///< direction of the transform
      PrecisionTriad                  precision{};                        ///< precision triad
      View<std::size_t, shapeExt>     shape{};                            ///< shape of the transform
      View<std::size_t, transformExt> axes{allAxes<transformExt>};        ///< axes of the transform
      Normalization                   normalization{Normalization::none}; ///< normalization
      Placement                       placement{Placement::outOfPlace};   ///< placement of the transform
      Type                            type{Type::separable};              ///< type of the transform
    };
  } // namespace dht

  /// @brief Namespace for discrete trigonometric transform
  namespace dtt
  {
    /// @brief DTT transform type
    enum class Type
    {
      dct1,       ///< Discrete Cosine Transform type I
      dct2,       ///< Discrete Cosine Transform type II
      dct3,       ///< Discrete Cosine Transform type III
      dct4,       ///< Discrete Cosine Transform type IV

      dst1,       ///< Discrete Sine Transform type I
      dst2,       ///< Discrete Sine Transform type II
      dst3,       ///< Discrete Sine Transform type III
      dst4,       ///< Discrete Sine Transform type IV

      dct = dct2, ///< default DCT type
      dst = dst2, ///< default DST type
    };

    /**
     * @brief DTT Parameters
     * @tparam shapeExt Extent of the shape, dynamic by default
     * @tparam transformExt Extent of the transform, dynamic by default
     * @tparam ttExt Extent of the types, dynamic by default
     */
    template<std::size_t shapeExt     = dynamicExtent,
             std::size_t transformExt = dynamicExtent,
             std::size_t ttExt        = dynamicExtent>
    struct Parameters : detail::TransformParametersBase<shapeExt, transformExt>
    {
      static_assert((shapeExt == dynamicExtent) || (shapeExt > 0), "shape rank must be greater than 0");
      static_assert((transformExt == dynamicExtent) || (transformExt > 0), "transform rank must be greater than 0");
      static_assert((shapeExt == dynamicExtent) || (transformExt == dynamicExtent) || (transformExt <= shapeExt),
                    "transform rank must be less than or equal to shape rank");
      static_assert((ttExt == dynamicExtent) || (ttExt == 1) || (transformExt == dynamicExtent || ttExt == transformExt),
                    "types rank must be 1 or equal to the number of axes");

      Direction                       direction{};                        ///< direction of the transform
      PrecisionTriad                  precision{};                        ///< precision triad
      View<std::size_t, shapeExt>     shape{};                            ///< shape of the transform
      View<std::size_t, transformExt> axes{allAxes<transformExt>};        ///< axes of the transform
      Normalization                   normalization{Normalization::none}; ///< normalization
      Placement                       placement{Placement::outOfPlace};   ///< placement of the transform
      View<Type, ttExt>               types{};                            ///< types of the transform, must have size 1 or size equal to the number of axes
    };
  } // namespace dtt
} // namespace afft

#endif /* AFFT_TRANSFORM_HPP */
