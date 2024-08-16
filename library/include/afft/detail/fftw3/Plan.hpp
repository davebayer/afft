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

#ifndef AFFT_DETAIL_FFTW3_PLAN_HPP
#define AFFT_DETAIL_FFTW3_PLAN_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../Plan.hpp"

namespace afft::detail::fftw3
{
  /// @brief The mkl plan implementation base class.
  class Plan : public afft::Plan
  {
    private:
      /// @brief Alias for the parent class.
      using Parent = afft::Plan;

    public:
      /// @brief Inherit constructor.
      using Parent::Parent;

      /// @brief Inherit assignment operator.
      using Parent::operator=;

      /// @brief Default destructor.
      virtual ~Plan() = default;

      /**
       * @brief Get the backend.
       * @return The backend.
       */
      [[nodiscard]] Backend getBackend() const noexcept override
      {
        return Backend::fftw3;
      }
    protected:
      /// @brief The plan deleter.
      template<afft::fftw3::Library library>
      struct PlanDeleter
      {
        /**
         * @brief Destroy the plan.
         * @param plan The plan.
         */
        void operator()(typename Lib<library>::Plan* plan) const noexcept
        {
          Lib<library>::destroyPlan(plan);
        }
      };

      /**
       * @brief Converts the planner flag to the FFTW3 flag.
       * @param flag The planner flag.
       * @return The FFTW3 flag.
       */
      [[nodiscard]] static constexpr unsigned
      makePlannerFlag(afft::fftw3::PlannerFlag flag)
      {
        validate(flag);

        switch (flag)
        {
        case afft::fftw3::PlannerFlag::estimate:
          return FFTW_ESTIMATE;
        case afft::fftw3::PlannerFlag::measure:
          return FFTW_MEASURE;
        case afft::fftw3::PlannerFlag::patient:
          return FFTW_PATIENT;
        case afft::fftw3::PlannerFlag::exhaustive:
          return FFTW_EXHAUSTIVE;
        case afft::fftw3::PlannerFlag::estimatePatient:
          return FFTW_ESTIMATE_PATIENT;
        default:
          cxx::unreachable();
        }
      }

      /**
       * @brief Converts the conserve memory flag to the FFTW3 flag.
       * @param conserveMemory The conserve memory flag.
       * @return The FFTW3 flag.
       */
      [[nodiscard]] static constexpr unsigned
      makeConserveMemoryFlag(bool conserveMemory) noexcept
      {
        return (conserveMemory) ? FFTW_CONSERVE_MEMORY : 0u;
      }

      /**
       * @brief Converts the wisdom only flag to the FFTW3 flag.
       * @param wisdomOnly The wisdom only flag.
       * @return The FFTW3 flag.
       */
      [[nodiscard]] static constexpr unsigned
      makeWisdomOnlyFlag(bool wisdomOnly) noexcept
      {
        return (wisdomOnly) ? FFTW_WISDOM_ONLY : 0u;
      }

      /**
       * @brief Converts the allow large generic flag to the FFTW3 flag.
       * @param allowLargeGeneric The allow large generic flag.
       * @return The FFTW3 flag.
       */
      [[nodiscard]] static constexpr unsigned
      makeAllowLargeGenericFlag(bool allowLargeGeneric) noexcept
      {
        return (allowLargeGeneric) ? FFTW_ALLOW_LARGE_GENERIC : 0u;
      }

      /**
       * @brief Converts the allow pruning flag to the FFTW3 flag.
       * @param allowPruning The allow pruning flag.
       * @return The FFTW3 flag.
       */
      [[nodiscard]] static constexpr unsigned
      makeAllowPruningFlag(bool allowPruning) noexcept
      {
        return (allowPruning) ? FFTW_ALLOW_PRUNING : 0u;
      }

      /**
       * @brief Converts the destructive flag to the FFTW3 flag.
       * @param isDestructive The destructive flag.
       * @return The FFTW3 flag.
       */
      [[nodiscard]] static constexpr unsigned
      makeDestructiveFlag(bool isDestructive) noexcept
      {
        return (isDestructive) ? FFTW_DESTROY_INPUT : FFTW_PRESERVE_INPUT;
      }

      /**
       * @brief Converts the alignment to the FFTW3 flag.
       * @param alignment The alignment.
       * @return The FFTW3 flag.
       */
      [[nodiscard]] static constexpr unsigned
      makeAlignmentFlag(Alignment alignment) noexcept
      {
        return (alignment < afft::Alignment{16}) ? FFTW_UNALIGNED : 0u;
      }

      /**
       * @brief Get the FFTW3 sign.
       * @return The FFTW3 sign.
       */
      [[nodiscard]] constexpr int getSign() const noexcept
      {
        return (mDesc.getDirection() == Direction::forward) ? FFTW_FORWARD : FFTW_BACKWARD;
      }

      /**
       * @brief Get the thread limit.
       * @return The thread limit.
       */
      [[nodiscard]] constexpr int getThreadLimit() const
      {
        return static_cast<int>(mDesc.getTargetDesc<Target::cpu>().threadLimit);
      }

      /**
       * @brief Gets the FFTW3 R2R kinds.
       * @tparam library The library.
       * @return The FFTW3 R2R kinds.
       */
      template<afft::fftw3::Library library>
      [[nodiscard]] MaxDimBuffer<typename Lib<library>::R2RKind>
      getR2RKinds() const
      {
        MaxDimBuffer<typename Lib<library>::R2RKind> r2rKinds{};

        const auto rank = mDesc.getTransformRank();

        switch (mDesc.getTransform())
        {
        case Transform::dht:
        {
          if (mDesc.getTransformDesc<Transform::dht>().type != dht::Type::separable)
          {
            throw Exception{Error::fftw3, "only separable DHT is supported"};
          }

          std::fill_n(r2rKinds.data, rank, FFTW_DHT);
          break;
        }
        case Transform::dtt:
        {
          const auto dttTypes = mDesc.getTransformDesc<Transform::dtt>().types;

          auto cvtDttType = [direction = mDesc.getDirection()](dtt::Type dttType)
          {
            switch (dttType)
            {
            case dtt::Type::dct1:
              return FFTW_REDFT00;
            case dtt::Type::dct2:
              return (direction == Direction::forward) ? FFTW_REDFT10 : FFTW_REDFT01;
            case dtt::Type::dct3:
              return (direction == Direction::forward) ? FFTW_REDFT01 : FFTW_REDFT10;
            case dtt::Type::dct4:
              return FFTW_REDFT11;
            case dtt::Type::dst1:
              return FFTW_RODFT00;
            case dtt::Type::dst2:
              return (direction == Direction::forward) ? FFTW_RODFT10 : FFTW_RODFT01;
            case dtt::Type::dst3:
              return (direction == Direction::forward) ? FFTW_RODFT01 : FFTW_RODFT10;
            case dtt::Type::dst4:
              return FFTW_RODFT11;
            default:
              cxx::unreachable();
            }
          };

          std::transform(dttTypes.begin(), dttTypes.begin() + rank, r2rKinds.data, cvtDttType);
          break;
        }
        default:
          throw Exception{Error::internal, "calling getR2RKinds() with invalid transform"};
        }

        return r2rKinds;
      }
  };
} // namespace afft::detail::fftw3

#endif /* AFFT_DETAIL_FFTW3_PLAN_HPP */
