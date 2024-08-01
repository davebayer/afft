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

#ifndef AFFT_DETAIL_FFTW3_PLAN_IMPL_HPP
#define AFFT_DETAIL_FFTW3_PLAN_IMPL_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "Lib.hpp"
#include "../cxx.hpp"
#include "../PlanImpl.hpp"
#include "../../exception.hpp"

namespace afft::detail::fftw3
{
  using namespace afft::cpu;

  /// @brief Size type for the FFTW library.
  using SizeT = std::ptrdiff_t;

  template<Precision prec>
  class PlanImpl : public detail::PlanImpl
  {
    static_assert(hasPrecision<prec>, "Unsupported precision");

    private:
      /// @brief Alias for the parent class.
      using Parent  = detail::PlanImpl;

      /// @brief Alias for the FFTW library plan.
      using Plan    = typename Lib<prec>::Plan;

      /// @brief Alias for the FFTW library real type.
      using R2RKind = typename Lib<prec>::R2RKind;

      /// @brief Alias for the FFTW library real type.
      using R       = typename Lib<prec>::Real;

      /// @brief Alias for the FFTW library complex type.
      using C       = typename Lib<prec>::Complex;

      /// @brief Alias for the FFTW library I/O dimension type.
      using IoDim   = typename Lib<prec>::IoDim;
    public:
      /// @brief Inherit the constructors from the parent class.
      using Parent::Parent;

      /**
       * @brief Constructor for the FFTW plan implementation.
       * @param config The configuration of the plan.
       */
      PlanImpl(const Config& config)
      : Parent(config)
      {
        const auto& commonParams = getConfig().getCommonParameters();
        const auto& cpuConfig    = getConfig().template getTargetConfig<Target::cpu>();

        const auto [rank, dims]               = makeDims(getConfig());
        const auto [howManyRank, howManyDims] = makeHowManyDims(getConfig());
        const auto flags                      = makeFlags(getConfig());

        std::size_t srcROrRiSize{};
        std::size_t srcISize{};
        std::size_t dstROrRISize{};
        std::size_t dstISize{};

        srcROrRiSize = getConfig().getSrcShapeVolume();
        dstROrRISize = getConfig().getDstShapeVolume();

        const auto [srcCmpl, dstCmpl] = getConfig().getSrcDstComplexity();

        switch (commonParams.complexFormat)
        {
        case ComplexFormat::interleaved:
          if (srcCmpl == Complexity::complex)
          {
            srcROrRiSize *= 2;
          }
          if (dstCmpl == Complexity::complex)
          {
            dstROrRISize *= 2;
          }
          break;
        case ComplexFormat::planar:
          if (srcCmpl == Complexity::complex)
          {
            srcISize = srcROrRiSize;
          }
          if (dstCmpl == Complexity::complex)
          {
            dstISize = dstROrRISize;
          }
          break;
        default:
          cxx::unreachable();
        }

        if (commonParams.placement == Placement::inPlace)
        {
          srcROrRiSize = std::max(srcROrRiSize, dstROrRISize);
          srcISize     = std::max(srcISize, dstISize);

          dstROrRISize = 0;
          dstISize     = 0;
        }

        const auto alignment = cpuConfig.alignment;

        AlignedUniquePtr<R[]> srcROrRI = (srcROrRiSize > 0) ? makeAlignedUniqueForOverwrite<R[]>(alignment, srcROrRiSize) : nullptr;
        AlignedUniquePtr<R[]> srcI     = (srcISize > 0)     ? makeAlignedUniqueForOverwrite<R[]>(alignment, srcISize) : nullptr;
        AlignedUniquePtr<R[]> dstROrRI = (dstROrRISize > 0) ? makeAlignedUniqueForOverwrite<R[]>(alignment, dstROrRISize) : nullptr;
        AlignedUniquePtr<R[]> dstI     = (dstISize > 0)     ? makeAlignedUniqueForOverwrite<R[]>(alignment, dstISize) : nullptr;

        ExecParam src{srcROrRI.get(), srcI.get()};
        ExecParam dst{(commonParams.placement == Placement::inPlace) ? srcROrRI.get() : dstROrRI.get(),
                      (commonParams.placement == Placement::inPlace) ? srcI.get() : dstI.get()};

        Lib<prec>::planWithNThreads(static_cast<int>(cpuConfig.threadLimit));

        Plan plan{};

        switch (getConfig().getTransform())
        {
        case Transform::dft:
        {
          const auto& dftConfig = getConfig().template getTransformConfig<Transform::dft>();
          const auto sign       = (getConfig().getTransformDirection() == Direction::forward)
                                    ? FFTW_FORWARD : FFTW_BACKWARD;

          switch (dftConfig.type)
          {
          case dft::Type::complexToComplex:
            if (commonParams.complexFormat == ComplexFormat::interleaved)
            {
              plan = Lib<prec>::planGuruC2C(rank,
                                            dims.data(),
                                            howManyRank,
                                            howManyDims.data(),
                                            src.getRealImagAs<C>(),
                                            dst.getRealImagAs<C>(),
                                            sign,
                                            flags);
            }
            else
            {
              plan = Lib<prec>::planGuruSplitC2C(rank,
                                                 dims.data(),
                                                 howManyRank,
                                                 howManyDims.data(),
                                                 src.getRealAs<R>(),
                                                 src.getImagAs<R>(),
                                                 dst.getRealAs<R>(),
                                                 dst.getImagAs<R>(),
                                                 flags);
            }
            break;
          case dft::Type::realToComplex:
            if (commonParams.complexFormat == ComplexFormat::interleaved)
            {
              plan = Lib<prec>::planGuruR2C(rank,
                                            dims.data(),
                                            howManyRank,
                                            howManyDims.data(),
                                            src.getRealAs<R>(),
                                            dst.getRealImagAs<C>(),
                                            flags);
            }
            else
            {
              plan = Lib<prec>::planGuruSplitR2C(rank,
                                                 dims.data(),
                                                 howManyRank,
                                                 howManyDims.data(),
                                                 src.getRealAs<R>(),
                                                 dst.getRealAs<R>(),
                                                 dst.getImagAs<R>(),
                                                 flags);
            }
            break;
          case dft::Type::complexToReal:
            if (commonParams.complexFormat == ComplexFormat::interleaved)
            {
              plan = Lib<prec>::planGuruC2R(rank,
                                            dims.data(),
                                            howManyRank,
                                            howManyDims.data(),
                                            src.getRealImagAs<C>(),
                                            dst.getRealAs<R>(),
                                            flags);
            }
            else
            {
              plan = Lib<prec>::planGuruSplitC2R(rank,
                                                 dims.data(),
                                                 howManyRank,
                                                 howManyDims.data(),
                                                 src.getRealAs<R>(),
                                                 src.getImagAs<R>(),
                                                 dst.getRealAs<R>(),
                                                 flags);
            }
            break;
          default:
            cxx::unreachable();
          }
          break;
        }
        case Transform::dtt:
        {
          const auto r2rKinds = makeR2RKinds(getConfig());

          plan = Lib<prec>::planGuruR2R(rank,
                                        dims.data(),
                                        howManyRank,
                                        howManyDims.data(),
                                        src.getRealAs<R>(),
                                        dst.getRealAs<R>(),
                                        r2rKinds.data(),
                                        flags);

          break;
        }
        default:
          cxx::unreachable();
        }

        if (plan == nullptr)
        {
          throw BackendError{Backend::fftw3, "failed to create plan"};
        }

        mPlan.reset(plan);
      }

      /**
       * @brief Implementation of the plan execution on the CPU.
       * @param src The source buffer.
       * @param dst The destination buffer.
       * @param params The execution parameters.
       */
      void executeImpl(ExecParam src, ExecParam dst, const afft::cpu::ExecutionParameters&) override
      {
        const auto& commonParams = getConfig().getCommonParameters();
        const auto direction     = getConfig().getTransformDirection();

        switch (getConfig().getTransform())
        {
        case Transform::dft:
        {
          const auto& dftConfig = getConfig().template getTransformConfig<Transform::dft>();

          switch (dftConfig.type)
          {
          case dft::Type::complexToComplex:
            if (commonParams.complexFormat == ComplexFormat::interleaved)
            {
              Lib<prec>::executeC2C(mPlan.get(),
                                    src.getRealImagAs<C>(),
                                    dst.getRealImagAs<C>());
            }
            else
            {
              if (direction == Direction::forward)
              {     
                Lib<prec>::executeSplitC2C(mPlan.get(),
                                           src.getRealAs<R>(),
                                           src.getImagAs<R>(),
                                           dst.getRealAs<R>(),
                                           dst.getImagAs<R>());
              }
              else
              {
                Lib<prec>::executeSplitC2C(mPlan.get(),
                                           src.getImagAs<R>(),
                                           src.getRealAs<R>(),
                                           dst.getImagAs<R>(),
                                           dst.getRealAs<R>());
              } 
            }
            break;
          case dft::Type::realToComplex:
            if (commonParams.complexFormat == ComplexFormat::interleaved)
            {
              Lib<prec>::executeR2C(mPlan.get(),
                                    src.getRealAs<R>(),
                                    dst.getRealImagAs<C>());
            }
            else
            {
              Lib<prec>::executeSplitR2C(mPlan.get(),
                                         src.getRealAs<R>(),
                                         dst.getRealAs<R>(),
                                         dst.getImagAs<R>());
            }
            break;
          case dft::Type::complexToReal:
            if (commonParams.complexFormat == ComplexFormat::interleaved)
            {
              Lib<prec>::executeC2R(mPlan.get(),
                                    src.getRealImagAs<C>(),
                                    dst.getRealAs<R>());
            }
            else
            {
              Lib<prec>::executeSplitC2R(mPlan.get(),
                                         src.getRealAs<R>(),
                                         src.getImagAs<R>(),
                                         dst.getRealAs<R>());
            }
            break;
          default:
            cxx::unreachable();
          }
          break;
        }
        case Transform::dtt:
          Lib<prec>::executeR2R(mPlan.get(),
                                src.getRealAs<R>(),
                                dst.getRealAs<R>());
          break;
        default:
          cxx::unreachable();
        }
      }
    protected:
    private:
      /**
       * @brief Deleter for the FFTW plan.
       */
      struct Deleter
      {
        /**
         * @brief Destroys the FFTW plan.
         * @param plan The FFTW plan.
         */
        void operator()(Plan plan)
        {
          if (plan != nullptr)
          {
            Lib<prec>::destroyPlan(plan);
          }
        }
      };

      /**
       * @brief Converts the configuration to the FFTW dimensions.
       * @param config The configuration.
       * @return The FFTW dimensions.
       */
      [[nodiscard]] static constexpr std::tuple<int, MaxDimArray<IoDim>> makeDims(const Config& config)
      {
        const auto rank       = config.getTransformRank();
        const auto dims       = config.template getTransformDims<SizeT>();
        const auto srcStrides = config.template getTransformSrcStrides<SizeT>();
        const auto dstStrides = config.template getTransformDstStrides<SizeT>();

        MaxDimArray<IoDim> fftwDims{};

        for (std::size_t i{}; i < rank; ++i)
        {
          fftwDims[i] = IoDim{/* .n  = */ dims[i],
                              /* .is = */ srcStrides[i],
                              /* .os = */ dstStrides[i]};
        }

        return std::make_tuple(static_cast<int>(rank), fftwDims);
      }

      /**
       * @brief Converts the configuration to the FFTW howMany dimensions.
       * @param config The configuration.
       * @return The FFTW howMany dimensions.
       */
      [[nodiscard]] static constexpr std::tuple<int, MaxDimArray<IoDim>> makeHowManyDims(const Config& config)
      {
        const auto howManyRank       = config.getTransformHowManyRank();
        const auto howManyDims       = config.template getTransformHowManyDims<SizeT>();
        const auto howManySrcStrides = config.template getTransformHowManySrcStrides<SizeT>();
        const auto howManyDstStrides = config.template getTransformHowManyDstStrides<SizeT>();

        MaxDimArray<IoDim> fftwHowManyDims{};

        for (std::size_t i{}; i < howManyRank; ++i)
        {
          fftwHowManyDims[i] = IoDim{/* .n  = */ howManyDims[i],
                                     /* .is = */ howManySrcStrides[i],
                                     /* .os = */ howManyDstStrides[i]};
        }

        return std::make_tuple(static_cast<int>(howManyRank), fftwHowManyDims);
      };

      /**
       * @brief Converts the DTT types to the FFTW R2R kinds.
       * @param config The configuration.
       * @return The FFTW R2R kinds.
       */
      [[nodiscard]] static constexpr MaxDimArray<R2RKind> makeR2RKinds(const Config& config)
      {
        const auto direction = config.getTransformDirection();
        const auto rank      = config.getTransformRank();
        const auto dttTypes  = config.template getTransformConfig<Transform::dtt>().axisTypes;

        auto cvtDttType = [direction](dtt::Type dttType)
        {
          switch (dttType)
          {
          case dtt::Type::dct1: return FFTW_REDFT00;
          case dtt::Type::dct2: return (direction == Direction::forward) ? FFTW_REDFT10 : FFTW_REDFT01;
          case dtt::Type::dct3: return (direction == Direction::forward) ? FFTW_REDFT01 : FFTW_REDFT10;
          case dtt::Type::dct4: return FFTW_REDFT11;
          case dtt::Type::dst1: return FFTW_RODFT00;
          case dtt::Type::dst2: return (direction == Direction::forward) ? FFTW_RODFT10 : FFTW_RODFT01;
          case dtt::Type::dst3: return (direction == Direction::forward) ? FFTW_RODFT01 : FFTW_RODFT10;
          case dtt::Type::dst4: return FFTW_RODFT11;
          default: cxx::unreachable();
          }
        };

        MaxDimArray<R2RKind> r2rKinds{};

        std::transform(dttTypes.begin(), dttTypes.begin() + rank, r2rKinds.begin(), cvtDttType);

        return r2rKinds;
      };

      /**
       * @brief Converts the configuration to the FFTW flags.
       * @param config The configuration.
       * @return The FFTW flags.
       */
      [[nodiscard]] static constexpr unsigned makeFlags(const Config& config)
      {
        const auto& commonParams = config.getCommonParameters();
        const auto& cpuConfig    = config.template getTargetConfig<Target::cpu>();

        unsigned flags{};

        switch (commonParams.initEffort)
        {
        case InitEffort::estimate:   flags |= FFTW_ESTIMATE;   break;
        case InitEffort::measure:    flags |= FFTW_MEASURE;    break;
        case InitEffort::patient:    flags |= FFTW_PATIENT;    break;
        case InitEffort::exhaustive: flags |= FFTW_EXHAUSTIVE; break;
        default:
          break;
        }

        switch (commonParams.workspacePolicy)
        {
        case WorkspacePolicy::minimal: flags |= FFTW_CONSERVE_MEMORY; break;
        default:
          break;
        }

        flags |= (commonParams.destroySource) ? FFTW_PRESERVE_INPUT : FFTW_DESTROY_INPUT;
        flags |= (cpuConfig.alignment < afft::Alignment{16}) ? FFTW_UNALIGNED : 0u;

        return flags;
      }

      std::unique_ptr<std::remove_pointer_t<Plan>, Deleter> mPlan; ///< The FFTW plan.
  };

  /**
   * @brief Factory function for creating a plan implementation.
   * @param config The configuration of the plan.
   * @return The plan implementation.
   */
  [[nodiscard]] inline std::unique_ptr<detail::PlanImpl> makePlanImpl(const Config& config)
  {
    switch (config.getTransform())
    {
    case Transform::dft:
    case Transform::dtt:
      break;
    default:
      throw makeException<std::runtime_error>("[FFTW3 error] Unsupported transform");
    }

    switch(config.getTransformPrecision().execution)
    {
    case Precision::f32:
      return std::make_unique<PlanImpl<Precision::f32>>(config);
    case Precision::f64:
      return std::make_unique<PlanImpl<Precision::f64>>(config);
#   if defined(AFFT_HAS_F80) && defined(AFFT_CPU_FFTW3_LONG_FOUND)
    case Precision::f80:
      return std::make_unique<PlanImpl<Precision::f80>>(config);
#   endif
#   if defined(AFFT_HAS_F128) && defined(AFFT_CPU_FFTW3_QUAD_FOUND)
    case Precision::f128:
      return std::make_unique<PlanImpl<Precision::f128>>(config);
#   endif
    default:
      throw std::runtime_error("[FFTW3 error] Unsupported precision");
    }
  }
} // namespace afft::detail::fftw3

#endif /* AFFT_DETAIL_FFTW3_PLAN_IMPL_HPP */
