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

#ifndef COMMON_HPP
#define COMMON_HPP

#include <afft/afft.h>
#include <afft/afft.hpp>

#include "convert.hpp"

// Precision
template<>
struct Convert<afft::Precision>
  : EnumConvertBase<afft::Precision, afft_Precision, afft_Error_invalidPrecision>
{
  static_assert(afft_Precision_bf16         == afft::Precision::bf16);
  static_assert(afft_Precision_f16          == afft::Precision::f16);
  static_assert(afft_Precision_f32          == afft::Precision::f32);
  static_assert(afft_Precision_f64          == afft::Precision::f64);
  static_assert(afft_Precision_f80          == afft::Precision::f80);
  static_assert(afft_Precision_f64f64       == afft::Precision::f64f64);
  static_assert(afft_Precision_f128         == afft::Precision::f128);

  static_assert(afft_Precision_float        == afft::Precision::_float);
  static_assert(afft_Precision_double       == afft::Precision::_double);
  static_assert(afft_Precision_longDouble   == afft::Precision::_longDouble);
  static_assert(afft_Precision_doubleDouble == afft::Precision::_doubleDouble);
  static_assert(afft_Precision_quad         == afft::Precision::_quad);
};

// Alignment
template<>
struct Convert<afft::Alignment>
  : EnumConvertBase<afft::Alignment, afft_Alignment, afft_Error_invalidAlignment>
{
  static_assert(afft_Alignment_simd128  == afft::Alignment::simd128);
  static_assert(afft_Alignment_simd256  == afft::Alignment::simd256);
  static_assert(afft_Alignment_simd512  == afft::Alignment::simd512);
  static_assert(afft_Alignment_simd1024 == afft::Alignment::simd1024);
  static_assert(afft_Alignment_simd2048 == afft::Alignment::simd2048);

  static_assert(afft_Alignment_sse      == afft::Alignment::sse);
  static_assert(afft_Alignment_sse2     == afft::Alignment::sse2);
  static_assert(afft_Alignment_sse3     == afft::Alignment::sse3);
  static_assert(afft_Alignment_sse4     == afft::Alignment::sse4);
  static_assert(afft_Alignment_sse4_1   == afft::Alignment::sse4_1);
  static_assert(afft_Alignment_sse4_2   == afft::Alignment::sse4_2);
  static_assert(afft_Alignment_avx      == afft::Alignment::avx);
  static_assert(afft_Alignment_avx2     == afft::Alignment::avx2);
  static_assert(afft_Alignment_avx512   == afft::Alignment::avx512);
  static_assert(afft_Alignment_neon     == afft::Alignment::neon);
  static_assert(afft_Alignment_sve      == afft::Alignment::sve);
};

// Complexity
template<>
struct Convert<afft::Complexity>
  : EnumConvertBase<afft::Complexity, afft_Complexity, afft_Error_invalidComplexity>
{
  static_assert(afft_Complexity_real    == afft::Complexity::real);
  static_assert(afft_Complexity_complex == afft::Complexity::complex);
};

// ComplexFormat
template<>
struct Convert<afft::ComplexFormat>
  : EnumConvertBase<afft::ComplexFormat, afft_ComplexFormat, afft_Error_invalidComplexFormat>
{
  static_assert(afft_ComplexFormat_interleaved == afft::ComplexFormat::interleaved);
  static_assert(afft_ComplexFormat_planar      == afft::ComplexFormat::planar);
};

// Direction
template<>
struct Convert<afft::Direction>
  : EnumConvertBase<afft::Direction, afft_Direction, afft_Error_invalidDirection>
{
  static_assert(afft_Direction_forward  == afft::Direction::forward);
  static_assert(afft_Direction_inverse  == afft::Direction::inverse);

  static_assert(afft_Direction_backward == afft::Direction::backward);
};

// Placement
template<>
struct Convert<afft::Placement>
  : EnumConvertBase<afft::Placement, afft_Placement, afft_Error_invalidPlacement>
{
  static_assert(afft_Placement_inPlace    == afft::Placement::inPlace);
  static_assert(afft_Placement_outOfPlace == afft::Placement::outOfPlace);

  static_assert(afft_Placement_notInPlace == afft::Placement::notInPlace);
};

// Transform
template<>
struct Convert<afft::Transform>
  : EnumConvertBase<afft::Transform, afft_Transform, afft_Error_invalidTransform>
{
  static_assert(afft_Transform_dft == afft::Transform::dft);
  static_assert(afft_Transform_dht == afft::Transform::dht);
  static_assert(afft_Transform_dtt == afft::Transform::dtt);
};

// Target
template<>
struct Convert<afft::Target>
  : EnumConvertBase<afft::Target, afft_Target, afft_Error_invalidTarget>
{
  static_assert(afft_Target_cpu == afft::Target::cpu);
  static_assert(afft_Target_gpu == afft::Target::gpu);
};

// Distribution
template<>
struct Convert<afft::Distribution>
  : EnumConvertBase<afft::Distribution, afft_Distribution, afft_Error_invalidDistribution>
{
  static_assert(afft_Distribution_spst == afft::Distribution::spst);
  static_assert(afft_Distribution_spmt == afft::Distribution::spmt);
  static_assert(afft_Distribution_mpst == afft::Distribution::mpst);
};

// Normalization
template<>
struct Convert<afft::Normalization>
  : EnumConvertBase<afft::Normalization, afft_Normalization, afft_Error_invalidNormalization>
{
  static_assert(afft_Normalization_none       == afft::Normalization::none);
  static_assert(afft_Normalization_orthogonal == afft::Normalization::orthogonal);
  static_assert(afft_Normalization_unitary    == afft::Normalization::unitary);
};

// Precision triad
template<>
struct Convert<afft::PrecisionTriad>
  : StructConvertBase<afft::PrecisionTriad, afft_PrecisionTriad>
{
  /**
   * @brief Convert from C to C++.
   * @param cValue C struct value.
   * @return C++ struct value.
   */
  [[nodiscard]] static constexpr afft::PrecisionTriad fromC(const afft_PrecisionTriad& cValue)
  {
    afft::PrecisionTriad cxxValue{};
    cxxValue.execution   = Convert<afft::Precision>::fromC(cValue.execution);
    cxxValue.source      = Convert<afft::Precision>::fromC(cValue.source);
    cxxValue.destination = Convert<afft::Precision>::fromC(cValue.destination);

    return cxxValue;
  }

  /**
   * @brief Convert from C++ to C.
   * @param cxxValue C++ struct value.
   * @return C struct value.
   */
  [[nodiscard]] static constexpr afft_PrecisionTriad toC(const afft::PrecisionTriad& cxxValue)
  {
    afft_PrecisionTriad cValue{};
    cValue.execution   = Convert<afft::Precision>::toC(cxxValue.execution);
    cValue.source      = Convert<afft::Precision>::toC(cxxValue.source);
    cValue.destination = Convert<afft::Precision>::toC(cxxValue.destination);

    return cValue;
  }
};

#endif /* COMMON_HPP */
