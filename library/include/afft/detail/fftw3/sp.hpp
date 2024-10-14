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

#ifndef AFFT_DETAIL_FFTW3_SP_HPP
#define AFFT_DETAIL_FFTW3_SP_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "../../Plan.hpp"

namespace afft::detail::fftw3::sp::cpu
{
  /**
   * @brief Create a fftw3 single process cpu plan implementation.
   * @param[in] desc Plan description.
   * @param[in] backendParams Backend parameters.
   * @return Plan implementation.
   */
  [[nodiscard]] std::unique_ptr<afft::Plan>
  makePlan(const Description&                  desc,
           const afft::cpu::BackendParameters& backendParams);
} // namespace afft::detail::fftw3::sp::cpu

#ifdef AFFT_HEADER_ONLY

#include "Plan.hpp"

namespace afft::detail::fftw3::sp::cpu
{
  /**
   * @class Plan
   * @brief The fftw3 single process cpu plan implementation.
   * @tparam library FFTW3 library type.
   */
  template<afft::fftw3::Library library>
  class Plan final : public fftw3::Plan<MpBackend::none, library>
  {
    private:
      /// @brief Alias for the parent class
      using Parent  = fftw3::Plan<MpBackend::none, library>;

      /// @brief Alias for the FFTW library real type.
      using R2RKind = typename Lib<library>::R2RKind;

      /// @brief Alias for the FFTW library real type.
      using R       = typename Lib<library>::Real;

      /// @brief Alias for the FFTW library complex type.
      using C       = typename Lib<library>::Complex;

      /// @brief Alias for the FFTW library I/O dimension type.
      using IoDim   = typename Lib<library>::IoDim;
    public:
      /// @brief Inherit constructor.
      using Parent::Parent;

      /**
       * @brief Constructor
       * @param desc The plan description
       */
      Plan(const Description&                  desc,
           const afft::cpu::BackendParameters& backendParams)
      : Parent{desc, backendParams}
      {
        Parent::mIsDestructive = (Parent::mBackendParams.allowDestructive ||
                                  Parent::mDesc.getPlacement() == Placement::inPlace);
        Parent::mDesc.getRefElemCounts(mSrcElemCount.data(), mDstElemCount.data());

        std::array<AlignedUniquePtr<R[]>, 2> src{};
        std::array<AlignedUniquePtr<R[]>, 2> dst{};

        if (Parent::mDesc.getComplexFormat() == ComplexFormat::interleaved)
        {
          const auto [srcComplexity, dstComplexity] = Parent::mDesc.getSrcDstComplexity();

          const std::size_t srcElemCount = mSrcElemCount[0] * ((srcComplexity == Complexity::real) ? 1 : 2);
          const std::size_t dstElemCount = mDstElemCount[0] * ((dstComplexity == Complexity::real) ? 1 : 2);

          if (Parent::mDesc.getPlacement() == Placement::outOfPlace)
          {
            src[0] = makeAlignedUniqueForOverwrite<R[]>(Parent::mDesc.getAlignment(), srcElemCount);
            dst[0] = makeAlignedUniqueForOverwrite<R[]>(Parent::mDesc.getAlignment(), dstElemCount);
          }
          else
          {
            src[0] = makeAlignedUniqueForOverwrite<R[]>(Parent::mDesc.getAlignment(), std::max(srcElemCount, dstElemCount));
          }
        }
        else
        {
          if (Parent::mDesc.getPlacement() == Placement::outOfPlace)
          {
            src[0] = makeAlignedUniqueForOverwrite<R[]>(Parent::mDesc.getAlignment(), mSrcElemCount[0]);
            src[1] = makeAlignedUniqueForOverwrite<R[]>(Parent::mDesc.getAlignment(), mSrcElemCount[1]);
            dst[0] = makeAlignedUniqueForOverwrite<R[]>(Parent::mDesc.getAlignment(), mDstElemCount[0]);
            dst[1] = makeAlignedUniqueForOverwrite<R[]>(Parent::mDesc.getAlignment(), mDstElemCount[1]);
          }
          else
          {
            src[0] = makeAlignedUniqueForOverwrite<R[]>(Parent::mDesc.getAlignment(), std::max(mSrcElemCount[0], mDstElemCount[0]));
            src[1] = makeAlignedUniqueForOverwrite<R[]>(Parent::mDesc.getAlignment(), std::max(mSrcElemCount[1], mDstElemCount[1]));
          }
        }

        Parent::setThreadLimit();
        Parent::setPlannerTimeLimit();

        const auto rank                = static_cast<int>(Parent::mDesc.getTransformRank());
        const auto howManyRank         = static_cast<int>(Parent::mDesc.getTransformHowManyRank());
        const auto [dims, howManyDims] = makeIoDims(Parent::mDesc);
        const auto flags               = makeFlags(Parent::mDesc, Parent::mBackendParams);

        typename Lib<library>::Plan* plan{};

        switch (Parent::mDesc.getTransform())
        {
        case Transform::dft:
        {
          switch (Parent::mDesc.template getTransformDesc<Transform::dft>().type)
          {
          case dft::Type::complexToComplex:
            if (Parent::mDesc.getComplexFormat() == ComplexFormat::interleaved)
            {
              plan = Lib<library>::planGuruC2C(rank,
                                               dims.data,
                                               howManyRank,
                                               howManyDims.data,
                                               reinterpret_cast<C*>(src[0].get()),
                                               reinterpret_cast<C*>((Parent::mDesc.getPlacement() == Placement::outOfPlace) ? dst[0].get() : src[0].get()),
                                               Parent::getSign(),
                                               flags);
            }
            else
            {
              plan = Lib<library>::planGuruSplitC2C(rank,
                                                    dims.data,
                                                    howManyRank,
                                                    howManyDims.data,
                                                    src[0].get(),
                                                    src[1].get(),
                                                    (Parent::mDesc.getPlacement() == Placement::outOfPlace) ? dst[0].get() : src[0].get(),
                                                    (Parent::mDesc.getPlacement() == Placement::outOfPlace) ? dst[1].get() : src[1].get(),
                                                    flags);
            }
            break;
          case dft::Type::realToComplex:
            if (Parent::mDesc.getComplexFormat() == ComplexFormat::interleaved)
            {
              plan = Lib<library>::planGuruR2C(rank,
                                               dims.data,
                                               howManyRank,
                                               howManyDims.data,
                                               src[0].get(),
                                               reinterpret_cast<C*>((Parent::mDesc.getPlacement() == Placement::outOfPlace) ? dst[0].get() : src[0].get()),
                                               flags);
            }
            else
            {
              plan = Lib<library>::planGuruSplitR2C(rank,
                                                    dims.data,
                                                    howManyRank,
                                                    howManyDims.data,
                                                    src[0].get(),
                                                    (Parent::mDesc.getPlacement() == Placement::outOfPlace) ? dst[0].get() : src[0].get(),
                                                    (Parent::mDesc.getPlacement() == Placement::outOfPlace) ? dst[1].get() : src[1].get(),
                                                    flags);
            }
            break;
          case dft::Type::complexToReal:
            if (Parent::mDesc.getComplexFormat() == ComplexFormat::interleaved)
            {
              plan = Lib<library>::planGuruC2R(rank,
                                               dims.data,
                                               howManyRank,
                                               howManyDims.data,
                                               reinterpret_cast<C*>(src[0].get()),
                                               (Parent::mDesc.getPlacement() == Placement::outOfPlace) ? dst[0].get() : src[0].get(),
                                               flags);
            }
            else
            {
              plan = Lib<library>::planGuruSplitC2R(rank,
                                                    dims.data,
                                                    howManyRank,
                                                    howManyDims.data,
                                                    src[0].get(),
                                                    src[1].get(),
                                                    (Parent::mDesc.getPlacement() == Placement::outOfPlace) ? dst[0].get() : src[0].get(),
                                                    flags);
            }
            break;
          default:
            cxx::unreachable();
          }
          break;
        }
        case Transform::dht:
        case Transform::dtt:
        {
          plan = Lib<library>::planGuruR2R(rank,
                                           dims.data,
                                           howManyRank,
                                           howManyDims.data,
                                           src[0].get(),
                                           (Parent::mDesc.getPlacement() == Placement::outOfPlace) ? dst[0].get() : src[0].get(),
                                           Parent::getR2RKinds().data,
                                           flags);

          break;
        }
        default:
          throw Exception{Error::fftw3, "unsupported transform"};
        }

        if (plan == nullptr)
        {
          throw Exception{Error::fftw3, "failed to create plan"};
        }

        mPlan.reset(plan);
      }

      /// @brief Destructor.
      ~Plan() = default;

      /// @brief Inherit assignment operator.
      using Parent::operator=;

      /**
       * @brief Get element count of the source buffers.
       * @return Element count of the source buffers.
       */
      [[nodiscard]] const std::size_t* getSrcElemCounts() const noexcept override
      {
        return mSrcElemCount.data();
      }

      /**
       * @brief Get element count of the destination buffers.
       * @return Element count of the destination buffers.
       */
      [[nodiscard]] const std::size_t* getDstElemCounts() const noexcept override
      {
        return mDstElemCount.data();
      }

      /**
       * @brief Execute the plan
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void executeBackendImpl(void* const* src, void* const* dst, const afft::cpu::ExecutionParameters&) override
      {
        switch (Parent::mDesc.getTransform())
        {
        case Transform::dft:
          switch (Parent::mDesc.template getTransformDesc<Transform::dft>().type)
          {
          case dft::Type::complexToComplex:
            if (Parent::mDesc.getComplexFormat() == ComplexFormat::interleaved)
            {
              Lib<library>::executeC2C(mPlan.get(),
                                       reinterpret_cast<C*>(src[0]),
                                       reinterpret_cast<C*>(dst[0]));
            }
            else
            {
              if (Parent::mDesc.getDirection() == Direction::forward)
              {     
                Lib<library>::executeSplitC2C(mPlan.get(),
                                              reinterpret_cast<R*>(src[0]),
                                              reinterpret_cast<R*>(src[1]),
                                              reinterpret_cast<R*>(dst[0]),
                                              reinterpret_cast<R*>(dst[1]));
              }
              else
              {
                Lib<library>::executeSplitC2C(mPlan.get(),
                                              reinterpret_cast<R*>(src[1]),
                                              reinterpret_cast<R*>(src[0]),
                                              reinterpret_cast<R*>(dst[1]),
                                              reinterpret_cast<R*>(dst[0]));
              } 
            }
            break;
          case dft::Type::realToComplex:
            if (Parent::mDesc.getComplexFormat() == ComplexFormat::interleaved)
            {
              Lib<library>::executeR2C(mPlan.get(),
                                       reinterpret_cast<R*>(src[0]),
                                       reinterpret_cast<C*>(dst[0]));
            }
            else
            {
              Lib<library>::executeSplitR2C(mPlan.get(),
                                            reinterpret_cast<R*>(src[0]),
                                            reinterpret_cast<R*>(dst[0]),
                                            reinterpret_cast<R*>(dst[1]));
            }
            break;
          case dft::Type::complexToReal:
            if (Parent::mDesc.getComplexFormat() == ComplexFormat::interleaved)
            {
              Lib<library>::executeC2R(mPlan.get(),
                                       reinterpret_cast<C*>(src[0]),
                                       reinterpret_cast<R*>(dst[0]));
            }
            else
            {
              Lib<library>::executeSplitC2R(mPlan.get(),
                                            reinterpret_cast<R*>(src[0]),
                                            reinterpret_cast<R*>(src[1]),
                                            reinterpret_cast<R*>(dst[0]));
            }
            break;
          default:
            cxx::unreachable();
          }
          break;
        case Transform::dht:
        case Transform::dtt:
          Lib<library>::executeR2R(mPlan.get(),
                                   reinterpret_cast<R*>(src[0]),
                                   reinterpret_cast<R*>(dst[0]));
          break;
        default:
          cxx::unreachable();
        }
      }

    private:
      /**
       * @brief Make the FFTW3 flags.
       * @param desc The plan description.
       * @param backendParams The backend parameters.
       * @return The FFTW3 flags.
       */
      [[nodiscard]] static constexpr unsigned
      makeFlags(const Desc&                         desc,
                const afft::cpu::BackendParameters& backendParams)
      {
        return Parent::makePlannerFlag(backendParams.fftw3.plannerFlag) |
               Parent::makeConserveMemoryFlag(backendParams.fftw3.conserveMemory) |
               Parent::makeWisdomOnlyFlag(backendParams.fftw3.wisdomOnly) |
               Parent::makeAllowLargeGenericFlag(backendParams.fftw3.allowLargeGeneric) |
               Parent::makeAllowPruningFlag(backendParams.fftw3.allowPruning) |
               Parent::makeDestructiveFlag(desc.getPlacement() == Placement::inPlace || backendParams.allowDestructive) |
               Parent::makeAlignmentFlag(desc.getAlignment());
      }

      /**
       * @brief Converts the configuration to the FFTW dimensions.
       * @param config The configuration.
       * @return The FFTW dimensions.
       */
      [[nodiscard]] static constexpr std::tuple<MaxDimBuffer<IoDim>, MaxDimBuffer<IoDim>>
      makeIoDims(const Desc& desc)
      {
        MaxDimBuffer<IoDim> dims{};
        MaxDimBuffer<IoDim> howManyDims{};

        const auto& memDesc = desc.getMemDesc<MemoryLayout::centralized>();

        const auto shape      = desc.getShape();
        const auto srcStrides = memDesc.getSrcStrides();
        const auto dstStrides = memDesc.getDstStrides();

        auto dimsIt     = dims.data;
        auto howManyIt  = howManyDims.data;

        std::for_each_n(desc.getTransformAxes(), desc.getTransformRank(), [&](auto axis)
        {
          dimsIt->n  = safeIntCast<std::ptrdiff_t>(shape[axis]);
          dimsIt->is = safeIntCast<std::ptrdiff_t>(srcStrides[axis]);
          dimsIt->os = safeIntCast<std::ptrdiff_t>(dstStrides[axis]);
          ++dimsIt;
        });

        std::for_each_n(desc.getTransformHowManyAxes(), desc.getTransformHowManyRank(), [&](auto axis)
        {
          howManyIt->n  = safeIntCast<std::ptrdiff_t>(shape[axis]);
          howManyIt->is = safeIntCast<std::ptrdiff_t>(srcStrides[axis]);
          howManyIt->os = safeIntCast<std::ptrdiff_t>(dstStrides[axis]);
          ++howManyIt;
        });

        return std::make_tuple(dims, howManyDims);
      }

      std::unique_ptr<typename Lib<library>::Plan, typename Parent::PlanDeleter> mPlan;            ///< The FFTW3 plan.
      std::array<std::size_t, 2>                                                 mSrcElemCount{};  ///< The number of elements in the source buffer
      std::array<std::size_t, 2>                                                 mDstElemCount{};  ///< The number of elements in the destination buffer
  };

  /**
   * @brief Create a fftw3 single process cpu plan implementation.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @return Plan implementation.
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE std::unique_ptr<afft::Plan>
  makePlan(const Description&                  desc,
           const afft::cpu::BackendParameters& backendParams)
  {
    const auto& descImpl = desc.get(DescToken::make());

    const auto prec = descImpl.getPrecision().execution;

# ifdef AFFT_FFTW3_HAS_FLOAT
    if (prec == Precision::_float)
    {
      return std::make_unique<Plan<afft::fftw3::Library::_float>>(desc, backendParams);
    }
# endif
# ifdef AFFT_FFTW3_HAS_DOUBLE
    if (prec == Precision::_double)
    {
      return std::make_unique<Plan<afft::fftw3::Library::_double>>(desc, backendParams);
    }
# endif
# ifdef AFFT_FFTW3_HAS_LONG
    if (prec == Precision::_longDouble)
    {
      return std::make_unique<Plan<afft::fftw3::Library::longDouble>>(desc, backendParams);
    }
# endif
# ifdef AFFT_FFTW3_HAS_QUAD
    if (prec == Precision::_quad)
    {
      return std::make_unique<Plan<afft::fftw3::Library::quad>>(desc, backendParams);
    }
# endif

    throw Exception{Error::fftw3, "unsupported precision"};
  }
} // namespace afft::detail::fftw3::sp::cpu

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_DETAIL_MKL_SP_HPP */