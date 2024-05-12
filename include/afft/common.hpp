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

#ifndef AFFT_COMMON_HPP
#define AFFT_COMMON_HPP

#include <cstddef>
#include <utility>

#include "backend.hpp"
#include "Span.hpp"

namespace afft
{
  /// @brief Maximum number of dimensions
  inline constexpr std::size_t maxDimCount{AFFT_MAX_DIM_COUNT};

  /**
   * @brief Non-owning const view of a contiguous sequence of objects.
   * @tparam T Type of the elements.
   * @tparam extent Number of elements in the span.
   */
  template<typename T, std::size_t extent = dynamicExtent>
  using View = Span<const T, extent>;

  /// @brief Precision of a floating-point number
  enum class Precision : std::uint8_t
  {
    bf16   = 0, ///< Google Brain's brain floating-point format
    f16    = 1, ///< IEEE 754 half-precision binary floating-point format
    f32    = 2, ///< IEEE 754 single-precision binary floating-point format
    f64    = 3, ///< IEEE 754 double-precision binary floating-point format
    f64f64 = 4, ///< double double precision (f128 simulated with two f64)
    f80    = 5, ///< x86 80-bit extended precision format
    f128   = 6, ///< IEEE 754 quadruple-precision binary floating-point format
  };

  /// @brief Alignment of a data type
  enum class Alignment : std::size_t {};

  /// @brief Complexity of a data type
  enum class Complexity : std::uint8_t
  {
    real,    ///< real
    complex, ///< complex
  };

  /// @brief Complex number format
  enum class ComplexFormat : std::uint8_t
  {
    interleaved, ///< interleaved complex format
    planar,      ///< planar complex format
  };

  /// @brief Direction of the transform
  enum class Direction : std::uint8_t
  {
    forward,            ///< forward transform
    inverse,            ///< inverse transform
    backward = inverse, ///< alias for inverse transform
  };

  /// @brief Placement of the transform
  enum class Placement : std::uint8_t
  {
    inPlace,                 ///< in-place transform
    outOfPlace,              ///< out-of-place transform
    notInPlace = outOfPlace, ///< alias for outOfPlace transform
  };

  /// @brief Transform type
  enum class Transform : std::uint8_t
  {
    dft, ///< Discrete Fourier Transform
    dht, ///< Discrete Hartley Transform
    dtt, ///< Discrete Trigonometric Transform
  };

  /// @brief Target
  enum class Target : std::uint8_t
  {
    cpu, ///< CPU target
    gpu, ///< GPU target
  };

  /// @brief Initialization effort
  enum class InitEffort : std::uint8_t
  {
    low,               ///< low effort initialization
    med,               ///< medium effort initialization
    high,              ///< high effort initialization
    max,               ///< maximum effort initialization
    
    estimate   = low,  ///< alias for low effort initialization in FFTW style
    measure    = med,  ///< alias for medium effort initialization in FFTW style
    patient    = high, ///< alias for high effort initialization in FFTW style
    exhaustive = max,  ///< alias for maximum effort initialization in FFTW style
  };

  /// @brief Initialization parameters
  struct InitParameters
  {
    BackendMask           backendMask{BackendMask::empty};                     ///< backend mask
    View<Backend>         backendInitOrder{};                                  ///< backend initialization order, empty view means default order for the target
    BackendSelectStrategy backendSelectStrategy{BackendSelectStrategy::first}; ///< backend select strategy
    InitEffort            initEffort{InitEffort::med};                         ///< plan initialization effort
  };

  /// @brief Normalization
  enum class Normalization : std::uint8_t
  {
    none,       ///< no normalization
    orthogonal, ///< 1/sqrt(N) normalization applied to both forward and inverse transform
    unitary,    ///< 1/N normalization applied to inverse transform
  };

  /// @brief Workspace policy
  enum class WorkspacePolicy : std::uint8_t
  {
    minimal,     ///< use as little workspace as possible
    performance, ///< perfer performance over workspace minimization
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
  inline constexpr View<std::size_t> allAxes{};

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

    /// @brief DFT parameters
    struct Parameters
    {
      Direction         direction{};                        ///< direction of the transform
      PrecisionTriad    precision{};                        ///< precision triad
      View<std::size_t> shape{};                            ///< shape of the transform
      View<std::size_t> axes{allAxes};                      ///< axes of the transform
      Normalization     normalization{Normalization::none}; ///< normalization
      Placement         placement{Placement::outOfPlace};   ///< placement of the transform
      Type              type{Type::complexToComplex};       ///< type of the transform
    };
  } // namespace dft

  /// @brief Namespace for discrete Hartley transform
  namespace dht
  {
    /// @brief DHT transform type
    struct Parameters
    {
      Direction         direction{};                        ///< direction of the transform
      PrecisionTriad    precision{};                        ///< precision triad
      View<std::size_t> shape{};                            ///< shape of the transform
      View<std::size_t> axes{allAxes};                      ///< axes of the transform
      Normalization     normalization{Normalization::none}; ///< normalization
      Placement         placement{Placement::outOfPlace};   ///< placement of the transform
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

    /// @brief DTT parameters
    struct Parameters
    {
      Direction         direction{};                        ///< direction of the transform
      PrecisionTriad    precision{};                        ///< precision triad
      View<std::size_t> shape{};                            ///< shape of the transform
      View<std::size_t> axes{allAxes};                      ///< axes of the transform
      Normalization     normalization{Normalization::none}; ///< normalization
      Placement         placement{Placement::outOfPlace};   ///< placement of the transform
      View<Type>        types{};                            ///< types of the transform, must have size 1 or size equal to the number of axes
    };
  } // namespace dtt

  /**
   * @brief Equality operator for PrecisionTriad
   * @param lhs left-hand side
   * @param rhs right-hand side
   * @return true if the precision triads are equal, false otherwise
   */
  [[nodiscard]] inline constexpr bool operator==(const PrecisionTriad& lhs, const PrecisionTriad& rhs)
  {
    return (lhs.execution == rhs.execution) &&
           (lhs.source == rhs.source) &&
           (lhs.destination == rhs.destination);
  }
} // namespace afft

#endif /* AFFT_COMMON_HPP */
