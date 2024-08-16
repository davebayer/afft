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
   * @param backendParams Backend parameters.
   * @param desc Plan description.
   * @return Plan implementation.
   */
  [[nodiscard]] std::unique_ptr<afft::Plan>
  makePlan(const Desc&                         desc,
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
  class Plan final : public fftw3::Plan
  {
    private:
      /// @brief Alias for the parent class
      using Parent = fftw3::Plan;

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
       * @param Desc The plan description
       */
      Plan(const Desc&                                desc,
           const afft::fftw3::cpu::BackendParameters& backendParams)
      : Parent{desc, Workspace::internal}
      {
        Lib<library>::planWithNThreads(getThreadLimit());

        const auto rank                = static_cast<int>(mDesc.getTransformRank());
        const auto howManyRank         = static_cast<int>(mDesc.getTransformHowManyRank());
        const auto [dims, howManyDims] = makeIoDims(mDesc);
        const auto flags               = makeFlags(mDesc, backendParams);

        typename Lib<library>::Plan* plan{};

        // TODO: allocate src and dst buffers
        std::array<void*, 2> src{};
        std::array<void*, 2> dst{};

        switch (mDesc.getTransform())
        {
        case Transform::dft:
        {
          switch (mDesc.getTransformDesc<Transform::dft>().type)
          {
          case dft::Type::complexToComplex:
            if (mDesc.getComplexFormat() == ComplexFormat::interleaved)
            {
              plan = Lib<library>::planGuruC2C(rank,
                                               dims.data,
                                               howManyRank,
                                               howManyDims.data,
                                               reinterpret_cast<C*>(src[0]),
                                               reinterpret_cast<C*>(dst[0]),
                                               getSign(),
                                               flags);
            }
            else
            {
              plan = Lib<library>::planGuruSplitC2C(rank,
                                                    dims.data,
                                                    howManyRank,
                                                    howManyDims.data,
                                                    reinterpret_cast<R*>(src[0]),
                                                    reinterpret_cast<R*>(src[1]),
                                                    reinterpret_cast<R*>(dst[0]),
                                                    reinterpret_cast<R*>(dst[1]),
                                                    flags);
            }
            break;
          case dft::Type::realToComplex:
            if (mDesc.getComplexFormat() == ComplexFormat::interleaved)
            {
              plan = Lib<library>::planGuruR2C(rank,
                                               dims.data,
                                               howManyRank,
                                               howManyDims.data,
                                               reinterpret_cast<R*>(src[0]),
                                               reinterpret_cast<C*>(dst[0]),
                                               flags);
            }
            else
            {
              plan = Lib<library>::planGuruSplitR2C(rank,
                                                    dims.data,
                                                    howManyRank,
                                                    howManyDims.data,
                                                    reinterpret_cast<R*>(src[0]),
                                                    reinterpret_cast<R*>(dst[0]),
                                                    reinterpret_cast<R*>(dst[1]),
                                                    flags);
            }
            break;
          case dft::Type::complexToReal:
            if (mDesc.getComplexFormat() == ComplexFormat::interleaved)
            {
              plan = Lib<library>::planGuruC2R(rank,
                                               dims.data,
                                               howManyRank,
                                               howManyDims.data,
                                               reinterpret_cast<C*>(src[0]),
                                               reinterpret_cast<R*>(dst[0]),
                                               flags);
            }
            else
            {
              plan = Lib<library>::planGuruSplitC2R(rank,
                                                    dims.data,
                                                    howManyRank,
                                                    howManyDims.data,
                                                    reinterpret_cast<R*>(src[0]),
                                                    reinterpret_cast<R*>(src[1]),
                                                    reinterpret_cast<R*>(dst[0]),
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
                                           reinterpret_cast<R*>(src[0]),
                                           reinterpret_cast<R*>(dst[0]),
                                           getR2RKinds<library>().data,
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
      [[nodiscard]] View<std::size_t> getSrcElemCounts() const noexcept override
      {
        if (mDesc.getComplexFormat() == ComplexFormat::interleaved)
        {
          return makeScalarView(mSrcElemCount[0]);
        }
        else
        {
          return mSrcElemCount;
        }
      }

      /**
       * @brief Get element count of the destination buffers.
       * @return Element count of the destination buffers.
       */
      [[nodiscard]] View<std::size_t> getDstElemCounts() const noexcept override
      {
        if (mDesc.getComplexFormat() == ComplexFormat::interleaved)
        {
          return makeScalarView(mDstElemCount[0]);
        }
        else
        {
          return mDstElemCount;
        }
      }

      /**
       * @brief Execute the plan
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void executeBackendImpl(View<void*> src, View<void*> dst, const afft::cpu::ExecutionParameters&) override
      {
        switch (mDesc.getTransform())
        {
        case Transform::dft:
          switch (mDesc.getTransformDesc<Transform::dft>().type)
          {
          case dft::Type::complexToComplex:
            if (mDesc.getComplexFormat() == ComplexFormat::interleaved)
            {
              Lib<library>::executeC2C(mPlan.get(),
                                       reinterpret_cast<C*>(src[0]),
                                       reinterpret_cast<C*>(dst[0]));
            }
            else
            {
              if (mDesc.getDirection() == Direction::forward)
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
            if (mDesc.getComplexFormat() == ComplexFormat::interleaved)
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
            if (mDesc.getComplexFormat() == ComplexFormat::interleaved)
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
      makeFlags(const Desc&                                desc,
                const afft::fftw3::cpu::BackendParameters& backendParams)
      {
        return makePlannerFlag(backendParams.plannerFlag) |
               makeConserveMemoryFlag(backendParams.conserveMemory) |
               makeWisdomOnlyFlag(backendParams.wisdomOnly) |
               makeAllowLargeGenericFlag(backendParams.allowLargeGeneric) |
               makeAllowPruningFlag(backendParams.allowPruning) |
               makeDestructiveFlag(desc.isDestructive()) |
               makeAlignmentFlag(desc.getAlignment());
      }

      /**
       * @brief Converts the configuration to the FFTW dimensions.
       * @param config The configuration.
       * @return The FFTW dimensions.
       */
      [[nodiscard]] static constexpr std::tuple<MaxDimBuffer<typename Lib<library>::IoDim>, MaxDimBuffer<typename Lib<library>::IoDim>>
      makeIoDims(const Desc& desc)
      {
        MaxDimBuffer<typename Lib<library>::IoDim> dims{};
        MaxDimBuffer<typename Lib<library>::IoDim> howManyDims{};

        const auto& memDesc = desc.getMemDesc<MemoryLayout::centralized>();

        const auto shape      = desc.getShape();
        const auto axes       = desc.getTransformAxes();
        const auto srcStrides = memDesc.getSrcStrides();
        const auto dstStrides = memDesc.getDstStrides();

        auto dimsIt     = dims.data;
        auto howManyIt  = howManyDims.data;

        for (std::size_t i{}; i < desc.getShapeRank(); ++i)
        {
          if (std::find(axes.begin(), axes.end(), static_cast<Axis>(i)) != axes.end())
          {
            dimsIt->n  = safeIntCast<std::ptrdiff_t>(shape[i]);
            dimsIt->is = safeIntCast<std::ptrdiff_t>(srcStrides[i]);
            dimsIt->os = safeIntCast<std::ptrdiff_t>(dstStrides[i]);
            ++dimsIt;
          }
          else
          {
            howManyIt->n  = safeIntCast<std::ptrdiff_t>(shape[i]);
            howManyIt->is = safeIntCast<std::ptrdiff_t>(srcStrides[i]);
            howManyIt->os = safeIntCast<std::ptrdiff_t>(dstStrides[i]);
            ++howManyIt;
          }
        }

        return std::make_tuple(dims, howManyDims);
      }

      std::unique_ptr<typename Lib<library>::Plan, PlanDeleter<library>> mPlan;            ///< The FFTW3 plan.
      std::array<std::size_t, 2>                                         mSrcElemCount{};  ///< The number of elements in the source buffer
      std::array<std::size_t, 2>                                         mDstElemCount{};  ///< The number of elements in the destination buffer
  };

  /**
   * @brief Create a fftw3 single process cpu plan implementation.
   * @param desc Plan description.
   * @param backendParams Backend parameters.
   * @return Plan implementation.
   */
  [[nodiscard]] AFFT_HEADER_ONLY_INLINE std::unique_ptr<afft::Plan>
  makePlan(const Desc&                         desc,
           const afft::cpu::BackendParameters& backendParams)
  {
    switch (backendParams.workspace)
    {
    case Workspace::any:
    case Workspace::internal:
      break;
    default:
      throw Exception{Error::fftw3, "unsupported workspace"};
    }

    const auto prec = desc.getPrecision().execution;

# ifdef AFFT_FFTW3_HAS_FLOAT
    if (prec == Precision::_float)
    {
      return std::make_unique<Plan<afft::fftw3::Library::_float>>(desc, backendParams.fftw3);
    }
# endif
# ifdef AFFT_FFTW3_HAS_DOUBLE
    if (prec == Precision::_double)
    {
      return std::make_unique<Plan<afft::fftw3::Library::_double>>(desc, backendParams.fftw3);
    }
# endif
# ifdef AFFT_FFTW3_HAS_LONG
    if (prec == Precision::_longDouble)
    {
      return std::make_unique<Plan<afft::fftw3::Library::longDouble>>(desc, backendParams.fftw3);
    }
# endif
# ifdef AFFT_FFTW3_HAS_QUAD
    if (prec == Precision::_quad)
    {
      return std::make_unique<Plan<afft::fftw3::Library::quad>>(desc, backendParams.fftw3);
    }
# endif

    throw Exception{Error::fftw3, "unsupported precision"};
  }
} // namespace afft::detail::fftw3::sp::cpu

#endif /* AFFT_HEADER_ONLY */

#endif /* AFFT_DETAIL_MKL_SP_HPP */