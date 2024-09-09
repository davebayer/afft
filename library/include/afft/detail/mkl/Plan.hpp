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

#ifndef AFFT_DETAIL_MKL_PLAN_HPP
#define AFFT_DETAIL_MKL_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "error.hpp"
#include "../Plan.hpp"

namespace afft::detail::mkl
{
  /// @brief The mkl plan implementation base class.
  template<MpBackend mpBackend, Target target>
  class Plan : public detail::Plan<mpBackend, target, Backend::mkl>
  {
    private:
      /// @brief Alias for the parent class.
      using Parent = detail::Plan<mpBackend, target, Backend::mkl>;

    public:
      /// @brief Inherit constructor.
      using Parent::Parent;

      /// @brief Default destructor.
      virtual ~Plan() = default;

      /// @brief Inherit assignment operator.
      using Parent::operator=;
      
    protected:
      /**
       * @brief Get the precision.
       * @return The precision.
       */
      [[nodiscard]] constexpr DFTI_CONFIG_VALUE getPrecision() const
      {
        switch (Parent::mDesc.getPrecision().execution)
        {
        case Precision::f32:
          return DFTI_SINGLE;
        case Precision::f64:
          return DFTI_DOUBLE;
        default:
          throw Exception{Error::mkl, "unsupported precision"};
        }
      }

      /**
       * @brief Get the forward domain.
       * @return The forward domain.
       */
      [[nodiscard]] constexpr DFTI_CONFIG_VALUE getForwardDomain() const
      {
        return (Parent::mDesc.template getTransformDesc<Transform::dft>().type == dft::Type::complexToComplex)
          ? DFTI_COMPLEX : DFTI_REAL;
      }

      /**
       * @brief Get the placement.
       * @return The placement.
       */
      [[nodiscard]] constexpr DFTI_CONFIG_VALUE getPlacement() const
      {
        return (Parent::mDesc.getPlacement() == Placement::inPlace)
          ? DFTI_INPLACE : DFTI_NOT_INPLACE;
      }

      /**
       * @brief Get the scale config parameter.
       * @return The scale config parameter.
       */
      [[nodiscard]] DFTI_CONFIG_PARAM getScaleConfigParam() const
      {
        return (Parent::mDesc.getDirection() == Direction::forward)
          ? DFTI_FORWARD_SCALE : DFTI_BACKWARD_SCALE;
      }
  };
} // namespace afft::detail::mkl

#endif /* AFFT_DETAIL_MKL_PLAN_HPP */
