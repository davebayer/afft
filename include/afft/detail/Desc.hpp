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

#ifndef AFFT_DETAIL_DESC_HPP
#define AFFT_DETAIL_DESC_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "include.hpp"
#endif

#include "ArchDesc.hpp"
#include "TransformDesc.hpp"

namespace afft::detail
{
  class Desc : public TransformDesc, public ArchDesc
  {
    public:
      /// @brief Default constructor is deleted.
      Desc() = delete;

      /// @brief Constructor.
      template<typename TransformParamsT, typename ArchParamsT>
      Desc(const TransformParamsT& transformParameters, const ArchParamsT& archParameters)
      : TransformDesc{transformParameters},
        ArchDesc{archParameters, getShapeRank()}
      {
        static_assert(isTransformParameters<TransformParamsT>, "TransformParamsT must be a TransformParameters type.");
        static_assert(isArchitectureParameters<ArchParamsT>, "ArchParamsT must be an ArchParameters type.");

        // TODO: check strides or set them to default values
      }

      /// @brief Copy constructor.
      Desc(const Desc&) = default;

      /// @brief Move constructor.
      Desc(Desc&&) = default;

      /// @brief Destructor.
      ~Desc() = default;

      /// @brief Copy assignment operator.
      Desc& operator=(const Desc&) = default;

      /// @brief Move assignment operator.
      Desc& operator=(Desc&&) = default;
    private:

  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_DESC_HPP */
