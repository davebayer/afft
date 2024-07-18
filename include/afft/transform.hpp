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
  enum class Transform : std::uint8_t
  {
    dft, ///< Discrete Fourier Transform
    dht, ///< Discrete Hartley Transform
    dtt, ///< Discrete Trigonometric Transform
  };

  /// @brief Direction of the transform
  enum class Direction : std::uint8_t
  {
    forward,            ///< forward transform
    inverse,            ///< inverse transform
    backward = inverse, ///< alias for inverse transform
  };

  /// @brief Normalization
  enum class Normalization : std::uint8_t
  {
    none,       ///< no normalization
    orthogonal, ///< 1/sqrt(N) normalization applied to both forward and inverse transform
    unitary,    ///< 1/N normalization applied to inverse transform
  };

  /// @brief Placement of the transform
  enum class Placement : std::uint8_t
  {
    inPlace,                 ///< in-place transform
    outOfPlace,              ///< out-of-place transform
    notInPlace = outOfPlace, ///< alias for outOfPlace transform
  };

  /**
   * @struct PrecisionTriad
   * @brief Precision triad
   */
  struct PrecisionTriad
  {
    Precision execution{};   ///< precision of the execution
    Precision source{};      ///< precision of the source data
    Precision destination{}; ///< precision of the destination data
  };

  /// @brief Named constant representing all axes (is empty view)
  inline constexpr View<Axis> allAxes{};
  
/**********************************************************************************************************************/
// Discrete Fourier Transform (DFT)
/**********************************************************************************************************************/
  namespace dft
  {
    enum class Type : std::uint8_t;
    struct Parameters;
  } // namespace dft

  /// @brief DFT transform type
  enum class dft::Type : std::uint8_t
  {
    complexToComplex,       ///< complex-to-complex transform
    realToComplex,          ///< real-to-complex transform
    complexToReal,          ///< complex-to-real transform

    c2c = complexToComplex, ///< alias for complex-to-complex transform
    r2c = realToComplex,    ///< alias for real-to-complex transform
    c2r = complexToReal,    ///< alias for complex-to-real transform
  };

  /// @brief DFT Parameters
  struct dft::Parameters
  {
    Direction      direction{};                                  ///< direction of the transform
    PrecisionTriad precision{};                                  ///< precision triad
    View<Size>     shape{};                                      ///< shape of the transform
    View<Axis>     axes{allAxes};                                ///< axes of the transform
    Normalization  normalization{Normalization::none};           ///< normalization
    Placement      placement{Placement::outOfPlace};             ///< placement of the transform
    bool           destructive{placement == Placement::inPlace}; ///< destructive transform
    Type           type{Type::complexToComplex};                 ///< type of the transform
  };

/**********************************************************************************************************************/
// Discrete Hartley Transform (DHT)
/**********************************************************************************************************************/
  namespace dht
  {
    enum class Type : std::uint8_t;
    struct Parameters;
  } // namespace dht

  /// @brief DHT transform type
  enum class dht::Type : std::uint8_t
  {
    separable, ///< separable DHT, computes the DHT along each axis independently
  };

  /// @brief DHT Parameters
  struct dht::Parameters
  {
    Direction      direction{};                                  ///< direction of the transform
    PrecisionTriad precision{};                                  ///< precision triad
    View<Size>     shape{};                                      ///< shape of the transform
    View<Axis>     axes{allAxes};                                ///< axes of the transform
    Normalization  normalization{Normalization::none};           ///< normalization
    Placement      placement{Placement::outOfPlace};             ///< placement of the transform
    bool           destructive{placement == Placement::inPlace}; ///< destructive transform
    Type           type{Type::separable};                        ///< type of the transform
  };

/**********************************************************************************************************************/
// Discrete Trigonomic Transform (DTT)
/**********************************************************************************************************************/
  namespace dtt
  {
    enum class Type : std::uint8_t;
    struct Parameters;
  } // namespace dtt

  /// @brief DTT transform type
  enum class dtt::Type : std::uint8_t
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

  /// @brief DTT Parameters
  struct dtt::Parameters
  {
    Direction      direction{};                                  ///< direction of the transform
    PrecisionTriad precision{};                                  ///< precision triad
    View<Size>     shape{};                                      ///< shape of the transform
    View<Axis>     axes{allAxes};                                ///< axes of the transform
    Normalization  normalization{Normalization::none};           ///< normalization
    Placement      placement{Placement::outOfPlace};             ///< placement of the transform
    bool           destructive{placement == Placement::inPlace}; ///< destructive transform
    View<Type>     types{};                                      ///< types of the transform, must have size 1 or size equal to the number of axes
  };

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
