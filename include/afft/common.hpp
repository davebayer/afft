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

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "detail/include.hpp"
#endif

#include "backend.hpp"
#include "Span.hpp"

AFFT_EXPORT namespace afft
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

  /// @brief Dynamic rank
  inline constexpr std::size_t dynamicRank{dynamicExtent};

  /// @brief Precision of a floating-point number
  enum class Precision : std::uint8_t
  {
    bf16,        ///< Google Brain's brain floating-point format
    f16,         ///< IEEE 754 half-precision binary floating-point format
    f32,         ///< IEEE 754 single-precision binary floating-point format
    f64,         ///< IEEE 754 double-precision binary floating-point format
    f64f64,      ///< double double precision (f128 simulated with two f64)
    f80,         ///< x86 80-bit extended precision format
    f128,        ///< IEEE 754 quadruple-precision binary floating-point format
    _longDouble, ///< Precision of long double, only for internal use
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

  /// @brief Distribution type
  enum class Distribution : std::uint8_t
  {
    spst,           ///< single process, single target
    spmt,           ///< single process, multiple targets
    mpst,           ///< multiple processes, single target
    
    single = spst,  ///< alias for single process, single target
    multi  = spmt,  ///< alias for single process, multiple targets
    mpi    = mpst,  ///< alias for multiple processes, single target
  };

  /// @brief Normalization
  enum class Normalization : std::uint8_t
  {
    none,       ///< no normalization
    orthogonal, ///< 1/sqrt(N) normalization applied to both forward and inverse transform
    unitary,    ///< 1/N normalization applied to inverse transform
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
  template<std::size_t tRank = dynamicRank>
  inline constexpr View<std::size_t, tRank> allAxes{};

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
     * @tparam sRank Rank of the shape, dynamic by default
     * @tparam tRank Rank of the transform, dynamic by default
     */
    template<std::size_t sRank = dynamicRank, std::size_t tRank = dynamicRank>
    struct Parameters
    {
      static_assert((sRank == dynamicRank) || (sRank > 0), "shape rank must be greater than 0");
      static_assert((tRank == dynamicRank) || (tRank > 0), "transform rank must be greater than 0");
      static_assert((sRank == dynamicRank) || (tRank == dynamicRank) || (tRank <= sRank),
                    "transform rank must be less than or equal to shape rank");

      Direction                direction{};                        ///< direction of the transform
      PrecisionTriad           precision{};                        ///< precision triad
      View<std::size_t, sRank> shape{};                            ///< shape of the transform
      View<std::size_t, tRank> axes{allAxes<tRank>};               ///< axes of the transform
      Normalization            normalization{Normalization::none}; ///< normalization
      Placement                placement{Placement::outOfPlace};   ///< placement of the transform
      Type                     type{Type::complexToComplex};       ///< type of the transform
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
     * @tparam sRank Rank of the shape, dynamic by default
     * @tparam tRank Rank of the transform, dynamic by default
     */
    template<std::size_t sRank = dynamicRank, std::size_t tRank = dynamicRank>
    struct Parameters
    {
      static_assert((sRank == dynamicRank) || (sRank > 0), "shape rank must be greater than 0");
      static_assert((tRank == dynamicRank) || (tRank > 0), "transform rank must be greater than 0");
      static_assert((sRank == dynamicRank) || (tRank == dynamicRank) || (tRank <= sRank),
                    "transform rank must be less than or equal to shape rank");

      Direction                direction{};                        ///< direction of the transform
      PrecisionTriad           precision{};                        ///< precision triad
      View<std::size_t, sRank> shape{};                            ///< shape of the transform
      View<std::size_t, tRank> axes{allAxes<tRank>};               ///< axes of the transform
      Normalization            normalization{Normalization::none}; ///< normalization
      Placement                placement{Placement::outOfPlace};   ///< placement of the transform
      Type                     type{Type::separable};              ///< type of the transform
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
     * @tparam sRank Rank of the shape, dynamic by default
     * @tparam tRank Rank of the transform, dynamic by default
     * @tparam ttRank Rank of the types, dynamic by default
     */
    template<std::size_t sRank = dynamicRank, std::size_t tRank = dynamicRank, std::size_t ttRank = dynamicRank>
    struct Parameters
    {
      static_assert((sRank == dynamicRank) || (sRank > 0), "shape rank must be greater than 0");
      static_assert((tRank == dynamicRank) || (tRank > 0), "transform rank must be greater than 0");
      static_assert((sRank == dynamicRank) || (tRank == dynamicRank) || (tRank <= sRank),
                    "transform rank must be less than or equal to shape rank");
      static_assert((ttRank == dynamicRank) || (ttRank == 1) || (tRank == dynamicRank || ttRank == tRank),
                    "types rank must be 1 or equal to the number of axes");

      Direction                direction{};                        ///< direction of the transform
      PrecisionTriad           precision{};                        ///< precision triad
      View<std::size_t, sRank> shape{};                            ///< shape of the transform
      View<std::size_t, tRank> axes{allAxes<tRank>};               ///< axes of the transform
      Normalization            normalization{Normalization::none}; ///< normalization
      Placement                placement{Placement::outOfPlace};   ///< placement of the transform
      View<Type, ttRank>       types{};                            ///< types of the transform, must have size 1 or size equal to the number of axes
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
