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

#ifndef AFFT_DETAIL_ROCFFT_PLAN_HPP
#define AFFT_DETAIL_ROCFFT_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "error.hpp"
#include "../../Plan.hpp"

namespace afft::detail::rocfft
{
  /// @brief The rocfft plan implementation base class.
  class Plan : public afft::Plan
  {
    private:
      /// @brief Alias for the parent class.
      using Parent = afft::Plan;

    public:
      /// @brief Inherit constructor.
      using Parent::Parent;

      /// @brief Default destructor.
      virtual ~Plan() = default;

      /// @brief Inherit assignment operator.
      using Parent::operator=;

      /**
       * @brief Get the backend.
       * @return The backend.
       */
      [[nodiscard]] Backend getBackend() const noexcept override
      {
        return Backend::rocfft;
      }      
    protected:
      /// @brief The rocFFT plan deleter.
      struct PlanDeleter
      {
        void operator()(rocfft_plan plan) const noexcept
        {
          rocfft_plan_destroy(plan);
        }
      };

      /// @brief The rocFFT plan description deleter.
      struct DescDeleter
      {
        void operator()(rocfft_plan_description desc) const noexcept
        {
          rocfft_plan_description_destroy(desc);
        }
      };

      /// @brief The rocFFT execution info deleter.
      struct ExecInfoDeleter
      {
        void operator()(rocfft_execution_info info) const noexcept
        {
          rocfft_execution_info_destroy(info);
        }
      };

      /**
       * @brief Get the rocFFT placement.
       * @return The rocFFT placement.
       */
      [[nodiscard]] constexpr rocfft_result_placement getRocfftPlacement() const noexcept
      {
        return (mDesc.getPlacement() == Placement::inPlace)
                 ? rocfft_placement_inplace : rocfft_placement_notinplace;
      }

      /**
       * @brief Get the rocFFT transform type.
       * @return The rocFFT transform type.
       */
      [[nodiscard]] constexpr rocfft_transform_type getRocfftTransformType() const
      {
        switch (mDesc.getTransformDesc<Transform::dft>().type)
        {
        case dft::Type::complexToComplex:
          return (mDesc.getDirection() == Direction::forward)
                   ? rocfft_transform_type_complex_forward : rocfft_transform_type_complex_inverse;
        case dft::Type::realToComplex:
          return rocfft_transform_type_real_forward;
        case dft::Type::complexToReal:
          return rocfft_transform_type_real_inverse;
        default:
          cxx::unreachable();
        }
      }

      /**
       * @brief Get the rocFFT precision.
       * @return The rocFFT precision.
       */
      [[nodiscard]] constexpr rocfft_precision getRocfftPrecision() const noexcept
      {
        switch (mDesc.getPrecision().execution)
        {
        case Precision::f16:
          return rocfft_precision_half;
        case Precision::f32:
          return rocfft_precision_single;
        case Precision::f64:
          return rocfft_precision_double;
        default:
          cxx::unreachable();
        }
      }

      /**
       * @brief Get the rocFFT dimensions.
       * @return The rocFFT dimensions.
       */
      [[nodiscard]] constexpr std::size_t getRocfftDimensions() const noexcept
      {
        return mDesc.getTransformRank();
      }

      /**
       * @brief Get the rocFFT lengths.
       * @return The rocFFT lengths.
       */
      [[nodiscard]] constexpr std::array<std::size_t, 3> getRocfftLengths() const
      {
        const auto dims = mDesc.getTransformDims();

        std::copy(dims.rbegin(), dims.rend(), dims.begin());

        return dims;
      }

      /**
       * @brief Get the number of transforms.
       * @return The number of transforms.
       */
      [[nodiscard]] constexpr std::size_t getRocfftNumberOfTransforms() const
      {
        return (mDesc.getTransformHowManyRank() == 0) ? 1 : mDesc.getTransformHowManyDims().first();
      }

      /**
       * @brief Get the rocFFT plan description.
       * @return The rocFFT plan description.
       */
      [[nodiscard]] constexpr std::unique_ptr<std::remove_pointer_t<rocfft_plan_description>, DescDeleter>
      getRocfftPlanDescription() const
      {
        std::unique_ptr<std::remove_pointer_t<rocfft_plan_description>, DescDeleter> rocfftDesc{};

        {
          rocfft_plan_description tmpRocfftDesc{};

          checkError(rocfft_plan_description_create(&tmpRocfftDesc));

          rocfftDesc.reset(tmpRocfftDesc);
        }

        rocfft_array_type srcArrayType{};
        rocfft_array_type dstArrayType{};

        const auto isComplexInterleaved = (mDesc.getComplexFormat() == ComplexFormat::interleaved);

        switch (mDesc.getTransformDesc<Transform::dft>().type)
        {
        case dft::Type::complexToComplex:
          srcArrayType = (isComplexInterleaved)
                           ? rocfft_array_type_complex_interleaved : rocfft_array_type_complex_planar;
          dstArrayType = (isComplexInterleaved)
                           ? rocfft_array_type_complex_interleaved : rocfft_array_type_complex_planar;
          break;
        case dft::Type::realToComplex:
          srcArrayType = rocfft_array_type_real;
          dstArrayType = (isComplexInterleaved)
                           ? rocfft_array_type_hermitian_interleaved : rocfft_array_type_hermitian_planar;
          break;
        case dft::Type::complexToReal:
          srcArrayType = (isComplexInterleaved)
                           ? rocfft_array_type_hermitian_interleaved : rocfft_array_type_hermitian_planar;
          dstArrayType = rocfft_array_type_real;
          break;
        default:
          cxx::unreachable();
        }

        std::array<std::size_t, 3> srcStrides{};
        std::array<std::size_t, 3> dstStrides{};

        const bool isSpst = (mDesc.getDistribution() == Distribution::spst);

        if (isSpst)
        {
          const auto& memLayout = mDesc.getMemoryLayout<Distribution::spst>();

          std::copy(memLayout.getSrcStrides().rbegin(), memLayout.getSrcStrides().rend(), srcStrides.begin());
          std::copy(memLayout.getDstStrides().rbegin(), memLayout.getDstStrides().rend(), dstStrides.begin());
        }

        checkError(rocfft_plan_description_set_data_layout(rocfftDesc.get(),
                                                           srcArrayType,
                                                           dstArrayType,
                                                           nullptr,
                                                           nullptr,
                                                           (isSpst) ? mDesc.getTransformRank() : 0,
                                                           (isSpst) ? srcStrides.data() : nullptr,
                                                           (mDesc.getTransformHowManyRank() == 0) ? 0 : mDesc.getTransformHowManyRankDims().first(),
                                                           (isSpst) ? mDesc.getTransformRank() : 0,
                                                           (isSpst) ? dstStrides.data() : nullptr,
                                                           (mDesc.getTransformHowManyRank() == 0) ? 0 : mDesc.getTransformHowManyRankDims().first()));

        return rocfftDesc;
      }

      std::unique_ptr<std::remove_pointer_t<rocfft_plan>, PlanDeleter>               mPlan{};     ///< The rocFFT plan.
      std::unique_ptr<std::remove_pointer_t<rocfft_execution_info>, ExecInfoDeleter> mExecInfo{}; ///< The rocFFT execution info.
  };
} // namespace afft::detail::rocfft

#endif /* AFFT_DETAIL_ROCFFT_PLAN_HPP */
