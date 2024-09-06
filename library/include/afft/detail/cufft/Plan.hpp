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

#ifndef AFFT_DETAIL_CUFFT_PLAN_HPP
#define AFFT_DETAIL_CUFFT_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "common.hpp"
#include "error.hpp"
#include "../cxx.hpp"
#include "../Plan.hpp"

namespace afft::detail::cufft
{
  /**
   * @brief cuFFT plan implementation.
   */
  template<MpBackend mpBackend>
  class Plan : public detail::Plan<mpBackend, Target::cuda, Backend::cufft>
  {
    private:
      /// @brief Alias for the parent class.
      using Parent = detail::Plan<mpBackend, Target::cuda, Backend::cufft>;

    public:
      /// @brief Inherit constructors from the parent class.
      using Parent::Parent;

      /// @brief Default destructor.
      virtual ~Plan() = default;

      /// @brief Inherit assignment operator from the parent class.
      using Parent::operator=;

    protected:
      static void makeCufftPlan(cufftHandle          handle,
                                const TransformDesc& transformDesc,
                                const MemDesc&       ,
                                std::size_t*         workspaceSize)
      {
        const auto rank  = static_cast<int>(transformDesc.getTransformRank());
        auto       n     = transformDesc.getShapeAs<SizeT>();
        const auto batch = (transformDesc.getTransformHowManyRank() == 1)
                             ? transformDesc.getTransformHowManyDimsAs<SizeT>()[0] : 1;
        checkError(cufftMakePlanMany64(handle,
                                       rank, n.data,
                                       nullptr, 1, 0,
                                       nullptr, 1, 0,
                                       CUFFT_C2C,
                                       batch,
                                       workspaceSize));

        // checkError(cufftXtMakePlanMany(handle,
        //                                rank, n.data(),
        //                                inembed.data(), istride, idist, inputType,
        //                                onembed.data(), ostride, odist, outputType,
        //                                batch, &workspaceSize, executionType));
      }

      /**
       * @brief Get the cuFFT direction from the plan description.
       * @return The cuFFT direction.
       */
      [[nodiscard]] int getDirection() const noexcept
      {
        switch (Parent::mDesc.getDirection())
        {
          case Direction::forward:
            return CUFFT_FORWARD;
          case Direction::backward:
            return CUFFT_INVERSE;
          default:
            cxx::unreachable();
        }
      }
  };
} // namespace afft::detail::cufft

#endif /* AFFT_DETAIL_CUFFT_PLAN_HPP */
