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

#ifndef AFFT_DETAIL_POCKETFFT_SPST_HPP
#define AFFT_DETAIL_POCKETFFT_SPST_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../include.hpp"
#endif

#include "PlanImpl.hpp"

namespace afft::detail::pocketfft::spst::cpu
{
  /**
   * @class PlanImpl
   * @tparam prec The precision of the data.
   * @brief Implementation of the plan for the spst cpu architecture using PocketFFT
   */
  template<typename PrecT>
  class PlanImpl final : public pocketfft::PlanImpl
  {
    private:
      /// @brief Alias for the parent class
      using Parent = pocketfft::PlanImpl;

      /// @brief Alias for the real type
      using R = PrecT;

      /// @brief Alias for the interleaved complex type
      using C = std::complex<PrecT>;
    public:
      /// @brief inherit constructors
      using Parent::Parent;

      /**
       * @brief Constructor
       * @param Desc The plan description
       */
      PlanImpl(const Desc& desc)
      : Parent{desc},
        mShape{getDesc().getShape().begin(), getDesc().getShape().end()},
        mSrcStrides(mShape.size()),
        mDstStrides(mShape.size()),
        mAxes{getDesc().getTransformAxes().begin(), getDesc().getTransformAxes().end()}
      {
        const auto& memLayout = getDesc().template getMemoryLayout<Distribution::spst>();

        std::transform(memLayout.getSrcStrides().begin(),
                       memLayout.getSrcStrides().end(),
                       mSrcStrides.begin(),
                       [this](const auto stride)
        {
          return safeIntCast<std::ptrdiff_t>(stride * getDesc().sizeOfSrcElem());
        });

        std::transform(memLayout.getDstStrides().begin(),
                       memLayout.getDstStrides().end(),
                       mDstStrides.begin(),
                       [this](const auto stride)
        {
          return safeIntCast<std::ptrdiff_t>(stride * getDesc().sizeOfDstElem());
        });
      }

      /// @brief Inherit assignment operator
      using Parent::operator=;

      /**
       * @brief Execute the plan
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void executeImpl(View<void*> src, View<void*> dst, const afft::spst::cpu::ExecutionParameters&) override
      {
        switch (getDesc().getTransform())
        {
        case Transform::dft:
          execDft(src.front(), dst.front());
          break;
        case Transform::dht:
          execDht(src.front(), dst.front());
          break;
        case Transform::dtt:
          execDtt(src.front(), dst.front());
          break;
        default:
          cxx::unreachable();
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
        const auto& dftDesc = getDesc().template getTransformDesc<Transform::dft>();

        const auto direction  = Parent::getDirection();
        const auto normFactor = getDesc().template getNormalizationFactor<R>();
        const auto nthreads   = getThreadCount();

        switch (dftDesc.type)
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
       * @brief Execute the DHT
       * @param src The source buffer
       * @param dst The destination buffer
       */
      void execDht(void* src, void* dst)
      {
        const auto& dhtDesc = getDesc().template getTransformDesc<Transform::dht>();

        const auto normFactor = getDesc().template getNormalizationFactor<R>();
        const auto nthreads   = getThreadCount();

        switch (dhtDesc.type)
        {
        case dht::Type::separable:
          safeCall([&, this]
          {
            ::pocketfft::r2r_separable_hartley(mShape,
                                               mSrcStrides,
                                               mDstStrides,
                                               mAxes,
                                               static_cast<R*>(src),
                                               static_cast<R*>(dst),
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
      void execDtt(void* src, void* dst)
      {
        static constexpr std::array dttTypes = {dtt::Type::dct1, dtt::Type::dct2, dtt::Type::dct3, dtt::Type::dct4,
                                                dtt::Type::dst1, dtt::Type::dst2, dtt::Type::dst3, dtt::Type::dst4};

        auto cvtDttType = [dir = getDesc().getDirection()](dtt::Type dttType) constexpr -> int
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

        const auto  axes    = getDesc().getTransformAxes();
        const auto& dttDesc = getDesc().template getTransformDesc<Transform::dtt>();

        auto normFactor = getDesc().template getNormalizationFactor<R>();

        const auto ortho    = (getDesc().getNormalization() == Normalization::orthogonal);
        const auto nthreads = getThreadCount();

        for (const auto dttType : dttTypes)
        {
          mAxes.clear();

          for (std::size_t i{}; i < getDesc().getTransformRank(); ++i)
          {
            if (dttDesc.types[i] == dttType)
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
                                 static_cast<R*>(src),
                                 static_cast<R*>(dst),
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
                                 static_cast<R*>(src),
                                 static_cast<R*>(dst),
                                 normFactor,
                                 ortho,
                                 nthreads);
              });
              break;
            default:
              cxx::unreachable();
            }

            normFactor = R{1.0};
          }
        }
      }

      [[nodiscard]] constexpr std::size_t getThreadCount() const
      {
        return static_cast<std::size_t>(getDesc().template getArchDesc<Target::cpu, Distribution::spst>().threadLimit);
      }

      ::pocketfft::shape_t  mShape{};      ///< The shape of the data
      ::pocketfft::stride_t mSrcStrides{}; ///< The stride of the source data
      ::pocketfft::stride_t mDstStrides{}; ///< The stride of the destination data
      ::pocketfft::shape_t  mAxes{};       ///< The axes to be transformed, valid for DFT, varies for DTT
  };

  /**
   * @brief Create a plan implementation.
   * @param desc Plan description.
   * @return Plan implementation.
   */
  [[nodiscard]] inline std::unique_ptr<pocketfft::PlanImpl>
  makePlanImpl(const Desc& desc)
  {
    // TODO: Adapt this and add DHT checks
    //
    // switch (config.getTransform())
    // {
    // case Transform::dft:
    // {
    //   const auto& dftConfig = config.getTransformConfig<Transform::dft>();

    //   switch (dftConfig.type)
    //   {
    //   case dft::Type::complexToComplex:
    //     if (commonParams.placement == Placement::inPlace)
    //     {
    //       if (!config.hasEqualStrides())
    //       {
    //         throw std::runtime_error{"Inplace transform requires equal strides"};
    //       }
    //     }
    //     break;
    //   case dft::Type::realToComplex:
    //   case dft::Type::complexToReal:
    //     if (commonParams.placement == Placement::inPlace)
    //     {
    //       throw std::runtime_error{"Inplace transform not supported"};
    //     }
    //     break;
    //   default:
    //     cxx::unreachable();
    //   }
    //   break;
    // }
    // case Transform::dtt:
    //   if (config.getCommonParameters().placement == Placement::inPlace)
    //   {
    //     if (!config.hasEqualStrides())
    //     {
    //       throw std::runtime_error{"Inplace transform requires equal strides"};
    //     }
    //   }
    //   break;
    // default:
    //   throw std::runtime_error{"Unsupported transform type"};
    // }

    if (!desc.hasUniformPrecision())
    {
      throw BackendError{Backend::pocketfft, "only same precision for execution, source and destination is supported"};
    }

    switch (const auto precision = desc.getPrecision().execution)
    {
      case Precision::_float:
        return std::make_unique<PlanImpl<float>>(desc);
      case Precision::_double:
        return std::make_unique<PlanImpl<double>>(desc);
      default:
        if (precision == Precision::_longDouble)
        {
          return std::make_unique<PlanImpl<long double>>(desc);
        }
        else
        {
          throw BackendError{Backend::pocketfft, "unsupported precision"};
        }
    }
  }
} // namespace afft::detail::pocketfft::spst::cpu

#endif /* AFFT_DETAIL_POCKETFFT_SPST_HPP */
