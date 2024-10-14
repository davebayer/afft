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
#include "type.hpp"

AFFT_EXPORT namespace afft
{
  /// @brief Transform type
  enum class Transform : ::afft_Transform
  {
    dft = afft_Transform_dft, ///< Discrete Fourier Transform
    dht = afft_Transform_dht, ///< Discrete Hartley Transform
    dtt = afft_Transform_dtt, ///< Discrete Trigonometric Transform
  };

  /// @brief Direction of the transform
  enum class Direction : ::afft_Direction
  {
    forward  = afft_Direction_forward,  ///< forward transform
    inverse  = afft_Direction_inverse,  ///< inverse transform
    backward = afft_Direction_backward, ///< alias for inverse transform
  };

  /// @brief Normalization
  enum class Normalization : ::afft_Normalization
  {
    none       = afft_Normalization_none,       ///< no normalization
    orthogonal = afft_Normalization_orthogonal, ///< 1/sqrt(N) normalization applied to both forward and inverse transform
    unitary    = afft_Normalization_unitary,    ///< 1/N normalization applied to inverse transform
  };

  /// @brief Placement of the transform
  enum class Placement : ::afft_Placement
  {
    outOfPlace = afft_Placement_outOfPlace, ///< out-of-place transform
    inPlace    = afft_Placement_inPlace,    ///< in-place transform
    notInPlace = afft_Placement_notInPlace, ///< alias for outOfPlace transform
  };

  /**
   * @struct PrecisionTriad
   * @brief Precision triad
   */
  struct PrecisionTriad
  {
    Precision execution{Precision::f32};   ///< execution precision
    Precision source{Precision::f32};      ///< source precision
    Precision destination{Precision::f32}; ///< destination precision
  };

  /// @brief Named constant representing all axes (is empty view)
  inline constexpr View<Axis> allAxes{};

/**********************************************************************************************************************/
// Discrete Fourier Transform (DFT)
/**********************************************************************************************************************/
  namespace dft
  {
    enum class Type : ::afft_dft_Type;
    struct Parameters;
  } // namespace dft

  /// @brief DFT transform type
  enum class dft::Type : ::afft_dft_Type
  {
    complexToComplex = afft_dft_Type_complexToComplex, ///< complex-to-complex transform
    realToComplex    = afft_dft_Type_realToComplex,    ///< real-to-complex transform
    complexToReal    = afft_dft_Type_complexToReal,    ///< complex-to-real transform
    c2c              = afft_dft_Type_c2c,              ///< alias for complex-to-complex transform
    r2c              = afft_dft_Type_r2c,              ///< alias for real-to-complex transform
    c2r              = afft_dft_Type_c2r,              ///< alias for complex-to-real transform
  };

  /// @brief DFT Parameters
  struct dft::Parameters
  {
    Direction      direction{};                        ///< direction of the transform
    PrecisionTriad precision{};                        ///< precision triad
    const Size*    shape{};                            ///< shape of the transform
    std::size_t    shapeRank{};                        ///< rank of the shape
    const Axis*    axes{};                             ///< axes of the transform
    std::size_t    axesRank{};                         ///< rank of the axes
    Normalization  normalization{Normalization::none}; ///< normalization
    Placement      placement{Placement::outOfPlace};   ///< placement of the transform
    Type           type{Type::complexToComplex};       ///< type of the transform
  };

/**********************************************************************************************************************/
// Discrete Hartley Transform (DHT)
/**********************************************************************************************************************/
  namespace dht
  {
    enum class Type : ::afft_dht_Type;
    struct Parameters;
  } // namespace dht

  /// @brief DHT transform type
  enum class dht::Type : ::afft_dht_Type
  {
    separable = afft_dht_Type_separable, ///< separable DHT, computes the DHT along each axis independently
  };

  /// @brief DHT Parameters
  struct dht::Parameters
  {
    Direction      direction{};                        ///< direction of the transform
    PrecisionTriad precision{};                        ///< precision triad
    const Size*    shape{};                            ///< shape of the transform
    std::size_t    shapeRank{};                        ///< rank of the shape
    const Axis*    axes{};                             ///< axes of the transform
    std::size_t    axesRank{};                         ///< rank of the axes
    Normalization  normalization{Normalization::none}; ///< normalization
    Placement      placement{Placement::outOfPlace};   ///< placement of the transform
    Type           type{Type::separable};              ///< type of the transform
  };

/**********************************************************************************************************************/
// Discrete Trigonomic Transform (DTT)
/**********************************************************************************************************************/
  namespace dtt
  {
    enum class Type : ::afft_dtt_Type;
    struct Parameters;
  } // namespace dtt

  /// @brief DTT transform type
  enum class dtt::Type : ::afft_dtt_Type
  {
    dct1 = afft_dtt_Type_dct1, ///< Discrete Cosine Transform type I
    dct2 = afft_dtt_Type_dct2, ///< Discrete Cosine Transform type II
    dct3 = afft_dtt_Type_dct3, ///< Discrete Cosine Transform type III
    dct4 = afft_dtt_Type_dct4, ///< Discrete Cosine Transform type IV
    dct5 = afft_dtt_Type_dct5, ///< Discrete Cosine Transform type V
    dct6 = afft_dtt_Type_dct6, ///< Discrete Cosine Transform type VI
    dct7 = afft_dtt_Type_dct7, ///< Discrete Cosine Transform type VII
    dct8 = afft_dtt_Type_dct8, ///< Discrete Cosine Transform type VIII
    dst1 = afft_dtt_Type_dst1, ///< Discrete Sine Transform type I
    dst2 = afft_dtt_Type_dst2, ///< Discrete Sine Transform type II
    dst3 = afft_dtt_Type_dst3, ///< Discrete Sine Transform type III
    dst4 = afft_dtt_Type_dst4, ///< Discrete Sine Transform type IV
    dst5 = afft_dtt_Type_dst5, ///< Discrete Sine Transform type V
    dst6 = afft_dtt_Type_dst6, ///< Discrete Sine Transform type VI
    dst7 = afft_dtt_Type_dst7, ///< Discrete Sine Transform type VII
    dst8 = afft_dtt_Type_dst8, ///< Discrete Sine Transform type VIII
    dct  = afft_dtt_Type_dct,  ///< alias for Discrete Cosine Transform type II
    dst  = afft_dtt_Type_dst,  ///< alias for Discrete Sine Transform type II
  };

  /// @brief DTT Parameters
  struct dtt::Parameters
  {
    Direction      direction{};                        ///< direction of the transform
    PrecisionTriad precision{};                        ///< precision triad
    const Size*    shape{};                            ///< shape of the transform
    std::size_t    shapeRank{};                        ///< rank of the shape
    const Axis*    axes{};                             ///< axes of the transform
    std::size_t    axesRank{};                         ///< rank of the axes
    Normalization  normalization{Normalization::none}; ///< normalization
    Placement      placement{Placement::outOfPlace};   ///< placement of the transform
    const Type*    types{};                            ///< types of the transform, must have size of transform rank
  };

/**********************************************************************************************************************/
// Transform variant
/**********************************************************************************************************************/
/// @brief Transform variant
using TransformParametersVariant = std::variant<std::monostate, dft::Parameters, dht::Parameters, dtt::Parameters>;

/**********************************************************************************************************************/
// Equality operators
/**********************************************************************************************************************/
  /**
   * @brief Equality operator for PrecisionTriad
   * @param lhs left-hand side
   * @param rhs right-hand side
   * @return true if the precision triads are equal, false otherwise
   */
  [[nodiscard]] constexpr bool operator==(const PrecisionTriad& lhs, const PrecisionTriad& rhs)
  {
    return (lhs.execution == rhs.execution) &&
           (lhs.source == rhs.source) &&
           (lhs.destination == rhs.destination);
  }
} // namespace afft

#endif /* AFFT_TRANSFORM_HPP */
