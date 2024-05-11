#ifndef AFFT_DETAIL_CPU_POCKETFFT_PLAN_IMPL_HPP
#define AFFT_DETAIL_CPU_POCKETFFT_PLAN_IMPL_HPP

#include <array>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include <pocketfft_hdronly.h>

#include "../../common.hpp"
#include "../../cxx.hpp"
#include "../../error.hpp"
#include "../../PlanImpl.hpp"
#include "../../Config.hpp"

namespace afft::detail::cpu::pocketfft
{
  // Check the types
  static_assert(std::is_same_v<::pocketfft::shape_t::value_type, std::size_t>,
                "afft requires std::size_t to be the same as pocketfft::shape_t::value_type");
  static_assert(std::is_same_v<::pocketfft::stride_t::value_type, std::ptrdiff_t>,
                "afft requires std::ptrdiff_t to be the same as pocketfft::stride_t::value_type");

  /**
   * @brief Safe call to a pocketfft function
   * @param fn The function to be invoked
   */
  template<typename Fn>
  void safeCall(Fn&& fn)
  {
    static_assert(std::is_invocable_v<decltype(fn)>, "fn must be invocable");

    try
    {
      std::invoke(fn);
    }
    catch (const std::exception& e)
    {
      throw makeException<std::runtime_error>(cformat("[PocketFFT error] %s", e.what()));
    }
  }

  /**
   * @class PlanImpl
   * @tparam prec The precision of the data.
   * @brief Implementation of the plan for the CPU using PocketFFT
   */
  template<Precision prec>
  class PlanImpl final : public detail::PlanImpl
  {
    private:
      /// @brief Alias for the parent class
      using Parent = detail::PlanImpl;

      /// @brief Alias for the real type
      using R = Real<prec>;

      /// @brief Alias for the interleaved complex type
      using C = Complex<R>;
    public:
      /// @brief inherit constructors
      using Parent::Parent;

      /**
       * @brief Constructor
       * @param config The configuration for the plan
       */
      PlanImpl(const Config& config)
      : Parent(config),
        mShape(getConfig().getShape().begin(), getConfig().getShape().end()),
        mSrcStrides(mShape.size()),
        mDstStrides(mShape.size()),
        mAxes(getConfig().getTransformAxes().begin(), getConfig().getTransformAxes().end())
      {
        std::transform(getConfig().getSrcStrides().begin(),
                       getConfig().getSrcStrides().end(),
                       mSrcStrides.begin(),
                       [this](const auto stride)
        {
          return safeIntCast<std::ptrdiff_t>(stride * getConfig().sizeOfSrcElem());
        });

        std::transform(getConfig().getDstStrides().begin(),
                       getConfig().getDstStrides().end(),
                       mDstStrides.begin(),
                       [this](const auto stride)
        {
          return safeIntCast<std::ptrdiff_t>(stride * getConfig().sizeOfDstElem());
        });
      }

      /// @brief Inherit assignment operator
      using Parent::operator=;

      /**
       * @brief Execute the plan
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void executeImpl(ExecParam src, ExecParam dst, const afft::cpu::ExecutionParameters&) override
      {
        if (src.isSplit() || dst.isSplit())
        {
          throw std::runtime_error("pocketfft does not support planar complex format");
        }

        switch (getConfig().getTransform())
        {
        case Transform::dft:
          execDft(src.getRealImag(), dst.getRealImag());
          break;
        case Transform::dtt:
          execDtt(src.getRealAs<R>(), dst.getRealAs<R>());
          break;
        default:
          throw makeException<std::runtime_error>("Unsupported transform type");
        }
      }
    protected:
    private:
      /**
       * @brief Execute the DFT
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void execDft(void* src, void* dst)
      {
        const auto& dftConfig = getConfig().template getTransformConfig<Transform::dft>();

        const auto direction  = (getConfig().getTransformDirection() == Direction::forward)
                                  ? ::pocketfft::FORWARD : ::pocketfft::BACKWARD;
        const auto nthreads   = static_cast<std::size_t>(getConfig().template getTargetConfig<Target::cpu>().threadLimit);
        const auto normFactor = getConfig().template getTransformNormFactor<prec>();

        switch (dftConfig.type)
        {
        case dft::Type::complexToComplex:
          safeCall([&, this]
          {
            ::pocketfft::c2c(mShape,
                             mSrcStrides,
                             mDstStrides,
                             mAxes,
                             direction,
                             static_cast<C*>(src),
                             static_cast<C*>(dst),
                             normFactor,
                             nthreads);
          });
          break;
        case dft::Type::realToComplex:
          safeCall([&, this]
          {
            ::pocketfft::c2r(mShape,
                             mSrcStrides,
                             mDstStrides,
                             mAxes,
                             direction,
                             static_cast<C*>(src),
                             static_cast<R*>(dst),
                             normFactor,
                             nthreads);
          });
          break;
        case dft::Type::complexToReal:
          safeCall([&, this]
          {
            ::pocketfft::r2c(mShape,
                             mSrcStrides,
                             mDstStrides,
                             mAxes,
                             direction,
                             static_cast<R*>(src),
                             static_cast<C*>(dst),
                             normFactor,
                             nthreads);
          });
          break;
        default:
          cxx::unreachable();
        }
      }

      /**
       * @brief Execute the DTT
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void execDtt(R* src, R* dst)
      {
        static constexpr std::array dttTypes = {dtt::Type::dct1, dtt::Type::dct2, dtt::Type::dct3, dtt::Type::dct4,
                                                dtt::Type::dst1, dtt::Type::dst2, dtt::Type::dst3, dtt::Type::dst4};

        auto cvtDttType = [dir = getConfig().getTransformDirection()](dtt::Type dttType) constexpr -> int
        {
          switch (dttType)
          {
          case dtt::Type::dct1: return (dir == Direction::forward) ? 1 : 1;
          case dtt::Type::dct2: return (dir == Direction::forward) ? 2 : 3;
          case dtt::Type::dct3: return (dir == Direction::forward) ? 3 : 2;
          case dtt::Type::dct4: return (dir == Direction::forward) ? 4 : 4;
          case dtt::Type::dst1: return (dir == Direction::forward) ? 1 : 1;
          case dtt::Type::dst2: return (dir == Direction::forward) ? 2 : 3;
          case dtt::Type::dst3: return (dir == Direction::forward) ? 3 : 2;
          case dtt::Type::dst4: return (dir == Direction::forward) ? 4 : 4;
          default:
            cxx::unreachable();
          }
        };

        const auto  axes      = getConfig().getTransformAxes();
        const auto& dttConfig = getConfig().template getTransformConfig<Transform::dtt>();

        auto normFactor = getConfig().template getTransformNormFactor<prec>();

        const auto ortho    = (getConfig().getCommonParameters().normalization == Normalization::orthogonal);
        const auto nthreads = static_cast<std::size_t>(getConfig().template getTargetConfig<Target::cpu>().threadLimit);

        for (const auto dttType : dttTypes)
        {
          mAxes.clear();

          for (std::size_t i{}; i < getConfig().getTransformRank(); ++i)
          {
            if (dttConfig.axisTypes[i] == dttType)
            {
              mAxes.push_back(axes[i]);
            }
          }

          if (!mAxes.empty())
          {
            switch (dttType)
            {
            case dtt::Type::dct1: case dtt::Type::dct2: case dtt::Type::dct3: case dtt::Type::dct4:
              safeCall([&, this]
              {
                ::pocketfft::dct(mShape,
                                 mSrcStrides,
                                 mDstStrides,
                                 mAxes,
                                 cvtDttType(dttType),
                                 src,
                                 dst,
                                 normFactor,
                                 ortho,
                                 nthreads);
              });
              break;
            case dtt::Type::dst1: case dtt::Type::dst2: case dtt::Type::dst3: case dtt::Type::dst4:
              safeCall([&, this]
              {
                ::pocketfft::dst(mShape,
                                 mSrcStrides,
                                 mDstStrides,
                                 mAxes,
                                 cvtDttType(dttType),
                                 src,
                                 dst,
                                 normFactor,
                                 ortho,
                                 nthreads);
              });
              break;
            default:
              throw makeException<std::runtime_error>("Unknown dtt type");
            }

            normFactor = R{1.0};
          }
        }
      }

      ::pocketfft::shape_t  mShape{};      ///< The shape of the data
      ::pocketfft::stride_t mSrcStrides{}; ///< The stride of the source data
      ::pocketfft::stride_t mDstStrides{}; ///< The stride of the destination data
      ::pocketfft::shape_t  mAxes{};       ///< The axes to be transformed, valid for DFT, varies for DTT
  };

  /**
   * @brief Factory function for creating a plan implementation
   * @param config The configuration for the plan implementation
   * @return The plan implementation
   */
  [[nodiscard]] inline std::unique_ptr<detail::PlanImpl> makePlanImpl(const Config& config)
  {
    const auto& commonParams = config.getCommonParameters();

    if (commonParams.complexFormat == ComplexFormat::planar)
    {
      throw makeException<std::runtime_error>("PocketFFT does not support planar complex format");
    }

    switch (config.getTransform())
    {
    case Transform::dft:
    {
      const auto& dftConfig = config.getTransformConfig<Transform::dft>();

      switch (dftConfig.type)
      {
      case dft::Type::complexToComplex:
        if (commonParams.placement == Placement::inPlace)
        {
          if (!config.hasEqualStrides())
          {
            throw makeException<std::runtime_error>("Inplace transform requires equal strides");
          }
        }
        break;
      case dft::Type::realToComplex:
      case dft::Type::complexToReal:
        if (commonParams.placement == Placement::inPlace)
        {
          throw makeException<std::runtime_error>("Inplace transform not supported");
        }
        break;
      default:
        cxx::unreachable();
      }
      break;
    }
    case Transform::dtt:
      if (config.getCommonParameters().placement == Placement::inPlace)
      {
        if (!config.hasEqualStrides())
        {
          throw makeException<std::runtime_error>("Inplace transform requires equal strides");
        }
      }
      break;
    default:
      throw makeException<std::runtime_error>("Unsupported transform type");
    }

    switch (config.getTransformPrecision().execution)
    {
#   ifdef AFFT_HAS_BF16
    case Precision::bf16:
      return std::make_unique<PlanImpl<Precision::bf16>>(config);
#   endif
#   ifdef AFFT_HAS_F16
    case Precision::f16:
      return std::make_unique<PlanImpl<Precision::f16>>(config);
#   endif
    case Precision::f32:
      return std::make_unique<PlanImpl<Precision::f32>>(config);
    case Precision::f64:
      return std::make_unique<PlanImpl<Precision::f64>>(config);
#   ifdef AFFT_HAS_F80
    case Precision::f80:
      return std::make_unique<PlanImpl<Precision::f80>>(config);
#   endif
#   ifdef AFFT_HAS_F128
    case Precision::f128:
      return std::make_unique<PlanImpl<Precision::f128>>(config);
#   endif
    default:
      throw makeException<std::runtime_error>("Unsupported precision");
    }
  }
} // namespace afft::detail::cpu::pocketfft

#endif /* AFFT_DETAIL_CPU_POCKETFFT_PLAN_IMPL_HPP */
