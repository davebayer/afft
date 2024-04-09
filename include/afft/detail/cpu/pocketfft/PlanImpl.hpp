#ifndef AFFT_DETAIL_CPU_POCKETFFT_PLAN_IMPL_HPP
#define AFFT_DETAIL_CPU_POCKETFFT_PLAN_IMPL_HPP

#if !__has_include(<pocketfft_hdronly.h>)
# error "PocketFFT header not found"
#endif

#include <array>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>

#include <pocketfft_hdronly.h>

#include "../../common.hpp"
#include "../../error.hpp"
#include "../../PlanImpl.hpp"
#include "../../Config.hpp"

namespace afft::detail::cpu::pocketfft
{
  // Check the types
  static_assert(std::same_as<::pocketfft::shape_t::value_type, std::size_t>);
  static_assert(std::same_as<::pocketfft::stride_t::value_type, std::ptrdiff_t>);

  /**
   * @brief Safe call to a pocketfft function
   * @param fn The function to be invoked
   */
  void safeCall(std::invocable auto&& fn)
  {
    try
    {
      std::invoke(fn);
    }
    catch (const std::exception& e)
    {
      throw makeException<std::runtime_error>(format("[PocketFFT error] {}", e.what()));
    }
  }

  /**
   * @class PlanImpl
   * @brief Implementation of the plan for the CPU using PocketFFT
   */
  class PlanImpl : public detail::PlanImpl
  {
    private:
      /// @brief Alias for the parent class
      using Parent = detail::PlanImpl;
    public:
      /// @brief inherit constructors
      using Parent::Parent;

      /**
       * @brief Constructor
       * @param config The configuration for the plan
       */
      PlanImpl(const Config& config) noexcept
      : Parent(checkConfig(config)),
        mShape(getConfig().dimsConfig.getShape().begin(), getConfig().dimsConfig.getShape().end()),
        mSrcStride(mShape.size()),
        mDstStride(mShape.size())
      {
        std::transform(getConfig().dimsConfig.getSrcStrides().begin(),
                       getConfig().dimsConfig.getSrcStrides().end(),
                       mSrcStride.begin(),
                       [this](const auto stride)
        {
          return safeIntCast<std::ptrdiff_t>(stride * getConfig().transformConfig.getSrcElemSizeOf());
        });

        std::transform(getConfig().dimsConfig.getDstStrides().begin(),
                       getConfig().dimsConfig.getDstStrides().end(),
                       mDstStride.begin(),
                       [this](const auto stride)
        {
          return safeIntCast<std::ptrdiff_t>(stride * getConfig().transformConfig.getDstElemSizeOf());
        });
      }

      /// @brief Inherit assignment operator
      using Parent::operator=;

      /**
       * @brief Execute the plan
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void execute(ExecParam src, ExecParam dst) override
      {
        if (!std::holds_alternative<void*>(src) || !std::holds_alternative<void*>(dst))
        {
          throw std::runtime_error("pocketfft does not support planar complex format");
        }

        switch (getConfig().transformConfig.getType())
        {
        case TransformType::dft: execDft(std::get<void*>(src), std::get<void*>(dst)); break;
        case TransformType::dtt: execDtt(std::get<void*>(src), std::get<void*>(dst)); break;
        default:                 throw makeException<std::runtime_error>("Unsupported transform type");
        }
      }
    protected:
    private:
      /**
       * @brief Check if the input configuration is supported
       * @param config The configuration to be checked
       * @return The configuration
       */
      static const Config& checkConfig(const Config& config)
      {
        switch (config.transformConfig.getType())
        {
        case TransformType::dft:
        {
          const auto& dftConfig = config.transformConfig.getConfig<TransformType::dft>();

          if (dftConfig.srcFormat == complexInterleaved && dftConfig.dstFormat == complexInterleaved)
          {
            if (config.commonParams.placement == Placement::inplace)
            {
              if (!config.dimsConfig.stridesEqual())
              {
                throw makeException<std::runtime_error>("Inplace transform requires equal strides");
              }
            }
          }
          else if (dftConfig.srcFormat == real && dftConfig.dstFormat == hermitianComplexInterleaved)
          {
            if (config.commonParams.placement == Placement::inplace)
            {
              throw makeException<std::runtime_error>("Inplace transform not supported");
            }
          }
          else if (dftConfig.srcFormat == hermitianComplexInterleaved && dftConfig.dstFormat == real)
          {
            if (config.commonParams.placement == Placement::inplace)
            {
              throw makeException<std::runtime_error>("Inplace transform not supported");
            }
          }
          else
          {
            throw makeException<std::runtime_error>("Unsupported DFT configuration");
          }
          break;
        }
        case TransformType::dtt:
          if (config.commonParams.placement == Placement::inplace)
          {
            if (!config.dimsConfig.stridesEqual())
            {
              throw makeException<std::runtime_error>("Inplace transform requires equal strides");
            }
          }
          break;
        default:
          throw makeException<std::runtime_error>("Unsupported transform type");
        }

        return config;
      }

      /**
       * @brief Execute the DFT
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void execDft(void* src, void* dst)
      {
        using enum dft::Format;

        const auto& precision    = getConfig().transformConfig.getPrecision();
        const auto& commonParams = getConfig().commonParams;
        const auto& dftConfig    = getConfig().transformConfig.getConfig<TransformType::dft>();
        const auto& cpuParams    = getConfig().targetConfig.getConfig<Target::cpu>();

        if (dftConfig.srcFormat == complexInterleaved && dftConfig.dstFormat == complexInterleaved)
        {
          switch (precision.execution)
          {
          case Precision::bf16:   this->c2c<Precision::bf16>(src, dst);   break;
          case Precision::f16:    this->c2c<Precision::f16>(src, dst);    break;
          case Precision::f32:    this->c2c<Precision::f32>(src, dst);    break;
          case Precision::f64:    this->c2c<Precision::f64>(src, dst);    break;
          case Precision::f64f64: this->c2c<Precision::f64f64>(src, dst); break;
          case Precision::f80:    this->c2c<Precision::f80>(src, dst);    break;
          case Precision::f128:   this->c2c<Precision::f128>(src, dst);   break;
          default:                throw makeExeption<std::runtime_error>("Unsupported precision");
          }
        }
        else if (dftConfig.srcFormat == real && dftConfig.dstFormat == hermitianComplexInterleaved)
        {
          switch (precision.execution)
          {
          case Precision::bf16:   this->r2c<Precision::bf16>(src, dst);   break;
          case Precision::f16:    this->r2c<Precision::f16>(src, dst);    break;
          case Precision::f32:    this->r2c<Precision::f32>(src, dst);    break;
          case Precision::f64:    this->r2c<Precision::f64>(src, dst);    break;
          case Precision::f64f64: this->r2c<Precision::f64f64>(src, dst); break;
          case Precision::f80:    this->r2c<Precision::f80>(src, dst);    break;
          case Precision::f128:   this->r2c<Precision::f128>(src, dst);   break;
          default:                throw makeExeption<std::runtime_error>("Unsupported precision");
          }
        }
        else if (dftConfig.srcFormat == hermitianComplexInterleaved && dftConfig.dstFormat == real)
        {
          switch (precision.execution)
          {
          case Precision::bf16:   this->c2r<Precision::bf16>(src, dst);   break;
          case Precision::f16:    this->c2r<Precision::f16>(src, dst);    break;
          case Precision::f32:    this->c2r<Precision::f32>(src, dst);    break;
          case Precision::f64:    this->c2r<Precision::f64>(src, dst);    break;
          case Precision::f64f64: this->c2r<Precision::f64f64>(src, dst); break;
          case Precision::f80:    this->c2r<Precision::f80>(src, dst);    break;
          case Precision::f128:   this->c2r<Precision::f128>(src, dst);   break;
          default:                throw makeExeption<std::runtime_error>("Unsupported precision");
          }
        }
        else
        {
          throw makeException<std::runtime_error>("Invalid dft compbination");
        }
      }

      /**
       * @brief Get the direction of the transform
       * @return The direction
       */
      [[nodiscard]] constexpr auto getDirection() const noexcept
      {
        return (getConfig().transformConfig.getDirection() == Direction::forward)
                 ? ::pocketfft::FORWARD : ::pocketfft::BACKWARD;
      }

      /**
       * @brief Execute the C2C transform
       * @tparam prec The precision of the data
       * @param src The source buffer
       * @param dst The destination buffer
       */
      template<Precision prec>
      void c2c(void* src, void* dst)
      {
        if constexpr (hasPrecision<prec>())
        {
          const auto normalize = getConfig().commonParams.normalize;
          const auto nthreads  = getConfig().transformConfig.getConfig<Target::cpu>().threadLimit;

          safeCall([this, &]
          {
            ::pocketfft::c2c(mShape,
                             mSrcStride,
                             mDstStride,
                             mAxes,
                             getDirection(),
                             reinterpret_cast<InterleavedComplex<prec>*>(src),
                             reinterpret_cast<InterleavedComplex<prec>*>(dst),
                             getConfig().transformConfig.getNormFactor<prec>(),
                             static_cast<std::size_t>(nthreads));
          });          
        }
        else
        {
          throw std::runtime_error("Not supported");
        }
      }

      /**
       * @brief Execute the R2C transform
       * @tparam prec The precision of the data
       * @param src The source buffer
       * @param dst The destination buffer
       */
      template<Precision prec>
      void r2c(void* src, void* dst)
      {
        if constexpr (hasPrecision<prec>())
        {
          const auto normalize = getConfig().commonParams.normalize;
          const auto nthreads  = getConfig().transformConfig.getConfig<Target::cpu>().threadLimit;

          safeCall([this, &]
          {
            ::pocketfft::r2c(mShape,
                             mSrcStride,
                             mDstStride,
                             mAxes,
                             getDirection(),
                             reinterpret_cast<Real<prec>*>(src),
                             reinterpret_cast<InterleavedComplex<prec>*>(dst),
                             getConfig().transformConfig.getNormFactor<prec>(),
                             static_cast<std::size_t>(nthreads));
          });
        }
        else
        {
          throw makeException<std::runtime_error>("Unsupported precision");
        }
      }

      /**
       * @brief Execute the C2R transform
       * @tparam prec The precision of the data
       * @param src The source buffer
       * @param dst The destination buffer
       */
      template<Precision prec>
      void c2r(void* src, void* dst)
      {
        if constexpr (hasPrecision<prec>())
        {
          const auto normalize = getConfig().commonParams.normalize;
          const auto nthreads  = getConfig().transformConfig.getConfig<Target::cpu>().threadLimit;

          safeCall([this, &]
          {
            ::pocketfft::c2r(mShape,
                            mSrcStride,
                            mDstStride,
                            mAxes,
                            getDirection(),
                            reinterpret_cast<InterleavedComplex<prec>*>(src),
                            reinterpret_cast<Real<prec>*>(dst),
                            getConfig().transformConfig.getNormFactor<prec>(),
                            static_cast<std::size_t>(nthreads));
          });
        }
        else
        {
          throw makeException<std::runtime_error>("Unsupported precision");
        }
      }

      /**
       * @brief Execute the DTT
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void execDtt(void* src, void* dst)
      {
        static constexpr std::array dttTypes = {dtt::Type::dct1, dtt::Type::dct2, dtt::Type::dct3, dtt::Type::dct4,
                                                dtt::Type::dst1, dtt::Type::dst2, dtt::Type::dst3, dtt::Type::dst4};

        const auto& precision = getConfig().commonParams.normalize;
        const auto& dttConfig = getConfig().transformConfig.getConfig<Target::cpu>().threadLimit;

        ::pocketfft::shape_t axes{};
        axes.reserve(getConfig().transformConfig.getRank());
        bool normalize = true;

        for (const auto dttType : dttTypes)
        {
          for (std::size_t i{}; i < getConfig().transformConfig.getRank(); ++i)
          {
            if (dttConfig.axisTypes[i] == dttType)
            {
              axes.push_back(getConfig().transformConfig.getAxes()[i]);
            }
          }

          if (!axes.empty())
          {
            switch (dttType)
            {
            case dtt::Type::dct1: case dtt::Type::dct2: case dtt::Type::dct3: case dtt::Type::dct4:
              switch (precision.execution)
              {
              case Precision::bf16:   this->dct<Precision::bf16>(src, dst, axes, dttType, normalize);   break;
              case Precision::f16:    this->dct<Precision::f16>(src, dst, axes, dttType, normalize);    break;
              case Precision::f32:    this->dct<Precision::f32>(src, dst, axes, dttType, normalize);    break;
              case Precision::f64:    this->dct<Precision::f64>(src, dst, axes, dttType, normalize);    break;
              case Precision::f64f64: this->dct<Precision::f64f64>(src, dst, axes, dttType, normalize); break;
              case Precision::f80:    this->dct<Precision::f80>(src, dst, axes, dttType, normalize);    break;
              case Precision::f128:   this->dct<Precision::f128>(src, dst, axes, dttType, normalize);   break;
              default:                throw makeException<std::runtime_error>("Unsupported precision");
              }
              break;
            case dtt::Type::dst1: case dtt::Type::dst2: case dtt::Type::dst3: case dtt::Type::dst4:
              switch (precision.execution)
              {
              case Precision::bf16:   this->dct<Precision::bf16>(src, dst, axes, dttType, normalize);   break;
              case Precision::f16:    this->dct<Precision::f16>(src, dst, axes, dttType, normalize);    break;
              case Precision::f32:    this->dst<Precision::f32>(src, dst, axes, dttType, normalize);    break;
              case Precision::f64:    this->dst<Precision::f64>(src, dst, axes, dttType, normalize);    break;
              case Precision::f64f64: this->dst<Precision::f64f64>(src, dst, axes, dttType, normalize); break;
              case Precision::f80:    this->dst<Precision::f80>(src, dst, axes, dttType, normalize);    break;
              case Precision::f128:   this->dst<Precision::f128>(src, dst, axes, dttType, normalize);   break;
              default:                throw makeException<std::runtime_error>("Unsupported precision");
              }
              break;
            default:
              throw makeException<std::runtime_error>("Unknown dtt type");
            }

            axes.clear();
            normalize = false;
          }
        }
      }

      /**
       * @brief Execute the DCT
       * @tparam prec The precision of the data
       * @param src The source buffer
       * @param dst The destination buffer
       * @param axes The axes to be transformed
       * @param dctType The type of the DCT
       * @param normalize Whether to normalize the result
       */
      template<Precision prec>
      void dct(void* src, void* dst, ::pocketfft::shape_t axes, dtt::Type dctType, bool normalize)
      {
        auto getType = [this, dctType]() -> int
        {
          const bool isForward = (getConfig().transformConfig.getDirection() == Direction::forward);

          switch (dctType)
          {
          case dtt::Type::dct1: return (isForward) ? 1 : 4;
          case dtt::Type::dct2: return (isForward) ? 2 : 3;
          case dtt::Type::dct3: return (isForward) ? 3 : 2;
          case dtt::Type::dct4: return (isForward) ? 4 : 1;
          default:
            throw makeException<std::runtime_error>("Invalid dct type");
          }
        };

        if constexpr (hasPrecision<prec>())
        {
          const auto nthreads = getConfig().targetConfig.getConfig<Target::cpu>().threadLimit;

          safeCall([this, &]
          {
            ::pocketfft::dct(mShape,
                             mSrcStride,
                             mDstStride,
                             axes,
                             getType(),
                             reinterpret_cast<Real<prec>*>(src),
                             reinterpret_cast<Real<prec>*>(dst),
                             (normalize) ? getConfig().transformConfig.getNormFactor<prec>() : Real<prec>(1.0),
                             (getConfig().commonParams.normalize == Normalize::orthogonal),
                             static_cast<std::size_t>(nthreads));
          });
        }
        else
        {
          throw makeException<std::runtime_error>("Unsupported precision");
        }
      }

      /**
       * @brief Execute the DST
       * @tparam prec The precision of the data
       * @param src The source buffer
       * @param dst The destination buffer
       * @param axes The axes to be transformed
       * @param dstType The type of the DST
       * @param normalize Whether to normalize the result
       */
      template<Precision prec>
      void dst(void* src, void* dst, ::pocketfft::shape_t axes, dtt::Type dstType, bool normalize)
      {
        auto getType = [this, dstType]() -> int
        {
          const bool isForward = (getConfig().transformConfig.getDirection() == Direction::forward);

          switch (dstType)
          {
          case dtt::Type::dst1: return (isForward) ? 1 : 4;
          case dtt::Type::dst2: return (isForward) ? 2 : 3;
          case dtt::Type::dst3: return (isForward) ? 3 : 2;
          case dtt::Type::dst4: return (isForward) ? 4 : 1;
          default:
            throw makeException<std::runtime_error>("Invalid dst type");
          }
        };
        
        if constexpr (hasPrecision<prec>())
        {
          const auto nthreads = getConfig().targetConfig.getConfig<Target::cpu>().threadLimit;

          safeCall([this, &]
          {
            ::pocketfft::dst(mShape,
                             mSrcStride,
                             mDstStride,
                             axes,
                             getType(),
                             reinterpret_cast<Real<prec>*>(src),
                             reinterpret_cast<Real<prec>*>(dst),
                             (normalize) ? getConfig().transformConfig.getNormFactor<prec>() : Real<prec>(1.0),
                             (getConfig().commonParams.normalize == Normalize::orthogonal),
                             static_cast<std::size_t>(nthreads));
          });
        }
        else
        {
          throw makeException<std::runtime_error>("Unsupported precision");
        }
      }

      ::pocketfft::shape_t  mShape{};     ///< The shape of the data
      ::pocketfft::stride_t mSrcStride{}; ///< The stride of the source data
      ::pocketfft::stride_t mDstStride{}; ///< The stride of the destination data
      ::pocketfft::shape_t  mAxes{};      ///< The axes to be transformed, valid only for DFT
  };
} // namespace afft::detail::cpu::pocketfft

#endif /* AFFT_DETAIL_CPU_POCKETFFT_PLAN_IMPL_HPP */
