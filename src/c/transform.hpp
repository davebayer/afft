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

#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP

#include <afft/afft.h>
#include <afft/afft.hpp>

#include "convert.hpp"

// DFT
template<>
struct Convert<afft::dft::Type>
  : EnumConvertBase<afft::dft::Type, afft_dft_Type, afft_Error_invalidDftType>
{
  static_assert(afft_dft_Type_complexToComplex == afft::dft::Type::complexToComplex);
  static_assert(afft_dft_Type_realToComplex    == afft::dft::Type::realToComplex);
  static_assert(afft_dft_Type_complexToReal    == afft::dft::Type::complexToReal);

  static_assert(afft_dft_Type_c2c              == afft::dft::Type::c2c);
  static_assert(afft_dft_Type_r2c              == afft::dft::Type::r2c);
  static_assert(afft_dft_Type_c2r              == afft::dft::Type::c2r);
};

template<std::size_t shapeRank, std::size_t transformRank>
struct Convert<afft::dft::Parameters<shapeRank, transformRank>>
  : StructConvertBase<afft::dft::Parameters<shapeRank, transformRank>, afft_dft_Parameters>
{
  /**
   * @brief Convert from C++ to C.
   * @param cxxType C++ type.
   * @return C type.
   */
  [[nodiscard]] static constexpr afft::dft::Parameters<> fromC(const afft_dft_Parameters& cType)
  {
    afft::dft::Parameters<> cxxType;
    cxxType.direction     = Convert<afft::dft::Type>::fromC(cType.direction);
    cxxType.precision     = Convert<afft::Precision>::fromC(cType.precision);
    cxxType.shape         = afft::View<std::size_t>{cType.shape, cType.shapeRank};
    cxxType.axes          = afft::View<std::size_t>{cType.axes, cType.axesRank};
    cxxType.normalization = Convert<afft::Normalization>::fromC(cType.normalization);
    cxxType.placement     = Convert<afft::Placement>::fromC(cType.placement);
    cxxType.type          = Convert<afft::dft::Type>::fromC(cType.type);

    return cxxType;
  }

  /**
   * @brief Convert from C to C++.
   * @param cxxType C++ type.
   * @return C type.
   */
  [[nodiscard]] static constexpr afft_dft_Parameters toC(const afft::dft::Parameters<shapeRank, transformRank>& cxxType)
  {
    afft_dft_Parameters cType;
    cType.direction     = Convert<afft::dft::Type>::toC(cxxType.direction);
    cType.precision     = Convert<afft::Precision>::toC(cxxType.precision);
    cType.shapeRank     = cxxType.shape.size();
    cType.shape         = cxxType.shape.data();
    cType.axesRank      = cxxType.axes.size();
    cType.axes          = cxxType.axes.data();
    cType.normalization = Convert<afft::Normalization>::toC(cxxType.normalization);
    cType.placement     = Convert<afft::Placement>::toC(cxxType.placement);
    cType.type          = Convert<afft::dft::Type>::toC(cxxType.type);

    return cType;
  }
};

// DHT
template<>
struct Convert<afft::dht::Type>
  : EnumConvertBase<afft::dht::Type, afft_dht_Type, afft_Error_invalidDhtType>
{
  static_assert(afft::dht::Type::separable == afft_dht_Type_separable);
};

template<std::size_t shapeRank, std::size_t transformRank>
struct Convert<afft::dht::Parameters<shapeRank, transformRank>>
  : StructConvertBase<afft::dht::Parameters<shapeRank, transformRank>, afft_dht_Parameters>
{
  /**
   * @brief Convert from C++ to C.
   * @param cxxType C++ type.
   * @return C type.
   */
  [[nodiscard]] static constexpr afft::dht::Parameters<> fromC(const afft_dht_Parameters& cType)
  {
    afft::dht::Parameters<> cxxType;
    cxxType.direction     = Convert<afft::dht::Type>::fromC(cType.direction);
    cxxType.precision     = Convert<afft::Precision>::fromC(cType.precision);
    cxxType.shape         = afft::View<std::size_t>{cType.shape, cType.shapeRank};
    cxxType.axes          = afft::View<std::size_t>{cType.axes, cType.axesRank};
    cxxType.normalization = Convert<afft::Normalization>::fromC(cType.normalization);
    cxxType.placement     = Convert<afft::Placement>::fromC(cType.placement);
    cxxType.type          = Convert<afft::dht::Type>::fromC(cType.type);

    return cxxType;
  }

  /**
   * @brief Convert from C to C++.
   * @param cxxType C++ type.
   * @return C type.
   */
  [[nodiscard]] static constexpr afft_dht_Parameters toC(const afft::dht::Parameters<shapeRank, transformRank>& cxxType)
  {
    afft_dht_Parameters cType;
    cType.direction     = Convert<afft::dht::Type>::toC(cxxType.direction);
    cType.precision     = Convert<afft::Precision>::toC(cxxType.precision);
    cType.shapeRank     = cxxType.shape.size();
    cType.shape         = cxxType.shape.data();
    cType.axesRank      = cxxType.axes.size();
    cType.axes          = cxxType.axes.data();
    cType.normalization = Convert<afft::Normalization>::toC(cxxType.normalization);
    cType.placement     = Convert<afft::Placement>::toC(cxxType.placement);
    cType.type          = Convert<afft::dht::Type>::toC(cxxType.type);

    return cType;
  }
};

// DTT
template<>
struct Convert<afft::dtt::Type>
  : EnumConvertBase<afft::dtt::Type, afft_dtt_Type, afft_Error_invalidDttType>
{
  static_assert(afft_dtt_Type_dct1 == afft::dtt::Type::dct1);
  static_assert(afft_dtt_Type_dct2 == afft::dtt::Type::dct2);
  static_assert(afft_dtt_Type_dct3 == afft::dtt::Type::dct3);
  static_assert(afft_dtt_Type_dct4 == afft::dtt::Type::dct4);
  static_assert(afft_dtt_Type_dst1 == afft::dtt::Type::dst1);
  static_assert(afft_dtt_Type_dst2 == afft::dtt::Type::dst2);
  static_assert(afft_dtt_Type_dst3 == afft::dtt::Type::dst3);
  static_assert(afft_dtt_Type_dst4 == afft::dtt::Type::dst4);

  static_assert(afft_dtt_Type_dct  == afft::dtt::Type::dct);
  static_assert(afft_dtt_Type_dst  == afft::dtt::Type::dst);
};

template<std::size_t shapeRank, std::size_t transformRank>
struct Convert<afft::dtt::Parameters<shapeRank, transformRank>>
  : StructConvertBase<afft::dtt::Parameters<shapeRank, transformRank>, afft_dtt_Parameters>
{
  /**
   * @brief Convert from C++ to C.
   * @param cxxType C++ type.
   * @return C type.
   */
  [[nodiscard]] static constexpr afft::dtt::Parameters<> fromC(const afft_dtt_Parameters& cType)
  {
    afft::dtt::Parameters<> cxxType;
    cxxType.direction     = Convert<afft::dtt::Type>::fromC(cType.direction);
    cxxType.precision     = Convert<afft::Precision>::fromC(cType.precision);
    cxxType.shape         = afft::View<std::size_t>{cType.shape, cType.shapeRank};
    cxxType.axes          = afft::View<std::size_t>{cType.axes, cType.axesRank};
    cxxType.normalization = Convert<afft::Normalization>::fromC(cType.normalization);
    cxxType.placement     = Convert<afft::Placement>::fromC(cType.placement);
    cxxType.types         = afft::View<afft::dtt::Type>{reinterpret_cast<const afft::dtt::Type*>(cType.types), cType.axesRank};

    for (const auto type : cxxType.types)
    {
      if (!afft::detail::isValid(type))
      {
        throw afft_Error_invalidDttType;
      }
    }

    return cxxType;
  }

  /**
   * @brief Convert from C to C++.
   * @param cxxType C++ type.
   * @return C type.
   */
  [[nodiscard]] static constexpr afft_dtt_Parameters toC(const afft::dtt::Parameters<shapeRank, transformRank>& cxxType)
  {
    afft_dtt_Parameters cType;
    cType.direction     = Convert<afft::dtt::Type>::toC(cxxType.direction);
    cType.precision     = Convert<afft::Precision>::toC(cxxType.precision);
    cType.shapeRank     = cxxType.shape.size();
    cType.shape         = cxxType.shape.data();
    cType.axesRank      = cxxType.axes.size();
    cType.axes          = cxxType.axes.data();
    cType.normalization = Convert<afft::Normalization>::toC(cxxType.normalization);
    cType.placement     = Convert<afft::Placement>::toC(cxxType.placement);
    cType.types         = Convert<afft::dtt::Type>::toC(reinterpret_cast<const afft_dtt_Type*>(cxxType.types.data()));

    for (const auto type : cxxType.types)
    {
      if (!afft::detail::isValid(type))
      {
        throw afft_Error_internal;
      }
    }

    return cType;
  }
};

#endif /* TRANSFORM_HPP */
