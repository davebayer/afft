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

#ifndef AFFT_DETAIL_PLAN_IMPL_HPP
#define AFFT_DETAIL_PLAN_IMPL_HPP

#include <cstddef>
#include <variant>

#include "Config.hpp"
#include "../gpu.hpp"

namespace afft::detail
{
  class ExecParam
  {
    public:
      constexpr ExecParam() = default;
      constexpr ExecParam(void* realOrRealImag, void* imag = nullptr) noexcept
      : mRealOrRealImag(realOrRealImag), mImag(imag)
      {}
      template<typename T>
      constexpr ExecParam(PlanarComplex<T> planarComplex) noexcept
      : mRealOrRealImag(planarComplex.real), mImag(planarComplex.imag)
      {}
      constexpr ExecParam(const ExecParam&) = default;
      constexpr ExecParam(ExecParam&&) = default;
      constexpr ~ExecParam() = default;

      constexpr ExecParam& operator=(const ExecParam&) = default;
      constexpr ExecParam& operator=(ExecParam&&) = default;

      [[nodiscard]] constexpr bool isSplit() const noexcept
      {
        return mImag != nullptr;
      }

      [[nodiscard]] constexpr void* getReal() const noexcept
      {
        return mRealOrRealImag;
      }

      template<typename T>
      [[nodiscard]] constexpr T* getRealAs() const noexcept
      {
        return static_cast<T*>(mRealOrRealImag);
      }

      [[nodiscard]] constexpr void* getRealImag() const noexcept
      {
        return mRealOrRealImag;
      }

      template<typename T>
      [[nodiscard]] constexpr T* getRealImagAs() const noexcept
      {
        return static_cast<T*>(mRealOrRealImag);
      }

      [[nodiscard]] constexpr void* getImag() const noexcept
      {
        return mImag;
      }

      template<typename T>
      [[nodiscard]] constexpr T* getImagAs() const noexcept
      {
        return static_cast<T*>(mImag);
      }
    protected:
    private:
      void* mRealOrRealImag{};
      void* mImag{};
  };

  /**
   * @class PlanImpl
   * @brief The abstract base class for all plan implementations
   */
  class PlanImpl
  {
    public:
      /// @brief The default constructor is deleted
      PlanImpl() = delete;

      /// @brief The copy constructor is deleted
      PlanImpl(const PlanImpl&) = delete;

      /// @brief The move constructor is defaulted
      PlanImpl(PlanImpl&&) = default;

      /// @brief The destructor is defaulted
      virtual ~PlanImpl() = default;

      /// @brief The copy assignment operator is deleted
      PlanImpl& operator=(const PlanImpl&) = delete;

      /// @brief The move assignment operator is defaulted
      PlanImpl& operator=(PlanImpl&&) = default;

      /**
       * @brief Get the configuration of the plan
       * @return const reference to the configuration of the plan
       */
      [[nodiscard]] constexpr const Config& getConfig() const noexcept
      {
        return mConfig;
      }

      /**
       * @brief Execute the plan
       * @param src the source buffer
       * @param dst the destination buffer
       */
      void execute(ExecParam src, ExecParam dst, const ExecutionParametersType auto& execParams)
      {
        executeImpl(src, dst, execParams);
      }

      /**
       * @brief Get the size of the workspace required for the plan
       * @return std::size_t the size of the workspace in bytes
       */
      virtual std::size_t getWorkspaceSize() const { return {}; }

      /**
       * @brief Check if the non-destructive transform is configured
       */
      constexpr void requireNonDestructiveTransform() const
      {
        if (mConfig.getCommonParameters().destroySource)
        {
          throw makeException<std::runtime_error>("Running a destructive transform on const input data.");
        }
      }

      /**
       * @brief Check if the out-of-place transform is configured
       */
      constexpr void requireOutOfPlaceTransform() const
      {
        if (mConfig.getCommonParameters().placement != Placement::outOfPlace)
        {
          throw makeException<std::runtime_error>("Running an in-place transform with out-of-place data.");
        }
      }

      /**
       * @brief Check if the in-place transform is configured
       */
      constexpr void requireInPlaceTransform() const
      {
        if (mConfig.getCommonParameters().placement != Placement::inPlace)
        {
          throw makeException<std::runtime_error>("Running an out-of-place transform with in-place data.");
        }
      }

      /**
       * @brief Check if the execution types are valid
       * @param srcPrec the source precision
       * @param srcCmpl the source complexity
       * @param dstPrec the destination precision
       * @param dstCmpl the destination complexity
       */
      constexpr void checkExecTypes(Precision srcPrec, Complexity srcCmpl, Precision dstPrec, Complexity dstCmpl) const
      {
        const auto& prec = mConfig.getTransformPrecision();

        if (srcPrec != prec.source)
        {
          throw makeException<std::invalid_argument>("Invalid source precision for transform");
        }
        
        if (dstPrec != prec.destination)
        {
          throw makeException<std::invalid_argument>("Invalid destination precision for transform");
        }

        switch (mConfig.getTransform())
        {
        case Transform::dft:
        {
          auto getFormatComplexity = [](dft::Type dftType)
          {
            switch (dftType)
            {
            case dft::Type::realToComplex:    return std::make_tuple(Complexity::real, Complexity::complex);
            case dft::Type::complexToReal:    return std::make_tuple(Complexity::complex, Complexity::real);
            case dft::Type::complexToComplex: return std::make_tuple(Complexity::complex, Complexity::complex);
            default:
              throw makeException<std::runtime_error>("Invalid DFT type");
            }
          };

          const auto& dftParams = mConfig.getTransformConfig<Transform::dft>();

          const auto [refSrcCmpl, refDstCmpl] = getFormatComplexity(dftParams.type);

          if (srcCmpl != refSrcCmpl)
          {
            throw makeException<std::invalid_argument>("Invalid source complexity for DFT transform");
          }

          if (dstCmpl != refDstCmpl)
          {
            throw makeException<std::invalid_argument>("Invalid destination complexity for DFT transform");
          }
          break;
        }
        case Transform::dtt:
          if (srcCmpl != Complexity::real)
          {
            throw makeException<std::invalid_argument>("Invalid source complexity for DTT transform");
          }

          if (dstCmpl != Complexity::real)
          {
            throw makeException<std::invalid_argument>("Invalid destination complexity for DTT transform");
          }
          break;
        default:
          throw makeException<std::runtime_error>("Invalid transform type");
        }
      }
    protected:
      /**
       * @brief Construct a new PlanImpl object
       * @param config the configuration of the plan
       */
      PlanImpl(const Config& config) noexcept
      : mConfig(config)
      {}

      /**
       * @brief Implementation of the plan execution on the CPU
       * @param src the source buffer
       * @param dst the destination buffer
       * @param execParams the CPU execution parameters
       */
      virtual void executeImpl(ExecParam, ExecParam, const afft::cpu::ExecutionParameters&)
      {
        throw makeException<std::runtime_error>("CPU execution is by currently selected implementation");
      }

      /**
       * @brief Implementation of the plan execution on the GPU
       * @param src the source buffer
       * @param dst the destination buffer
       * @param execParams the GPU execution parameters
       */
      virtual void executeImpl(ExecParam, ExecParam, const afft::gpu::ExecutionParameters&)
      {
        throw makeException<std::runtime_error>("GPU execution is by currently selected implementation");
      }

    private:
      Config mConfig; ///< The configuration of the plan
  };
} // namespace afft::detail

#endif /* AFFT_DETAIL_PLAN_IMPL_HPP */
